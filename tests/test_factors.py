"""Factor library tests: signs, ordering, warm-up and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlab.factors import (
    amihud,
    liquidity,
    momentum,
    short_term_reversal,
    trend_quality,
    volatility,
)
from factorlab.panel import to_long


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=n, freq="B")


def _assert_full_panel(series: pd.Series, wide: pd.DataFrame) -> None:
    assert len(series) == wide.shape[0] * wide.shape[1]
    expected = pd.MultiIndex.from_product(
        [wide.index, wide.columns], names=["date", "asset"]
    )
    assert series.index.equals(expected)


def test_momentum_orders_assets_by_past_return() -> None:
    dates = _dates(12)
    close = pd.DataFrame(
        {
            "WIN": 100.0 * 1.10 ** np.arange(12),
            "FLAT": np.full(12, 100.0),
            "LOSE": 100.0 * 0.97 ** np.arange(12),
        },
        index=dates,
    )
    mom = momentum(close, lookback=5, skip=0)
    _assert_full_panel(mom, close)

    # pct_change() is NaN on the first row, so the first `lookback` rows are NaN.
    assert mom.loc[dates[:5]].isna().all()
    assert mom.loc[dates[5]].notna().all()

    last = mom.loc[dates[-1]]
    assert last["WIN"] > 0.0
    assert last["FLAT"] == pytest.approx(0.0, abs=1e-12)
    assert last["LOSE"] < 0.0
    assert last["WIN"] > last["FLAT"] > last["LOSE"]

    expected = 1.10**5 - 1.0
    assert last["WIN"] == pytest.approx(expected, rel=1e-10)


def test_momentum_skip_and_no_lookahead() -> None:
    dates = _dates(20)
    rng = np.random.default_rng(1)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (20, 3)), axis=0)),
        index=dates,
        columns=list("ABC"),
    )
    mom = momentum(close, lookback=6, skip=2)

    perturbed = close.copy()
    perturbed.iloc[15] *= 1.5
    mom_perturbed = momentum(perturbed, lookback=6, skip=2)

    # A change at t=15 cannot affect a factor computed at t=14 or earlier.
    pd.testing.assert_series_equal(
        mom.loc[dates[:15]], mom_perturbed.loc[dates[:15]]
    )


def test_momentum_validation() -> None:
    close = pd.DataFrame({"A": [1.0, 2.0]}, index=_dates(2))
    with pytest.raises(ValueError, match="lookback"):
        momentum(close, lookback=0)
    with pytest.raises(ValueError, match="skip"):
        momentum(close, skip=-1)
    with pytest.raises(ValueError, match="positive"):
        momentum(pd.DataFrame({"A": [0.0, 1.0]}, index=_dates(2)))


def test_reversal_sign() -> None:
    dates = _dates(10)
    returns = pd.DataFrame(
        {"UP": np.full(10, 0.01), "DOWN": np.full(10, -0.01)}, index=dates
    )
    rev = short_term_reversal(returns, window=5)
    _assert_full_panel(rev, returns)
    assert rev.loc[dates[:4]].isna().all()

    last = rev.loc[dates[-1]]
    assert last["UP"] == pytest.approx(-0.05)
    assert last["DOWN"] == pytest.approx(0.05)
    assert last["UP"] < 0.0 < last["DOWN"]


def test_volatility_is_negative_and_low_vol_scores_higher() -> None:
    dates = _dates(40)
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(
        {
            "NOISY": rng.normal(0.0, 0.05, 40),
            "QUIET": rng.normal(0.0, 0.002, 40),
        },
        index=dates,
    )
    vol = volatility(returns, window=20)
    _assert_full_panel(vol, returns)
    assert vol.loc[dates[:19]].isna().all()

    last = vol.loc[dates[-1]]
    assert last["NOISY"] < last["QUIET"]
    expected = -returns["NOISY"].iloc[-20:].std(ddof=1)
    assert last["NOISY"] == pytest.approx(expected)


def test_liquidity_turnover_zscore() -> None:
    dates = _dates(2)
    assets = ["A", "B", "C", "D"]
    volume = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]], index=dates, columns=assets
    )
    mcap = pd.DataFrame(10.0, index=dates, columns=assets)
    liq = liquidity(volume, mcap)
    _assert_full_panel(liq, volume)

    first = liq.loc[dates[0]]
    assert first.is_monotonic_increasing
    assert first.mean() == pytest.approx(0.0, abs=1e-12)
    # Constant turnover has zero cross-sectional variance -> undefined z-score.
    assert liq.loc[dates[1]].isna().all()


def test_amihud_monotone_in_price_impact() -> None:
    dates = _dates(30)
    returns = pd.DataFrame(
        {"IMPACT": np.full(30, 0.02), "LIQUID": np.full(30, 0.002)},
        index=dates,
    )
    volume = pd.DataFrame(
        {"IMPACT": np.full(30, 1000.0), "LIQUID": np.full(30, 1000.0)},
        index=dates,
    )
    illiq = amihud(returns, volume, window=21)
    _assert_full_panel(illiq, returns)
    assert illiq.loc[dates[:20]].isna().all()

    last = illiq.loc[dates[-1]]
    assert last["IMPACT"] > last["LIQUID"] > 0.0
    assert last["IMPACT"] == pytest.approx(0.02 / 1000.0)

    more_volume = volume.copy()
    more_volume["IMPACT"] = 10000.0
    illiq_deep = amihud(returns, more_volume, window=21)
    assert illiq_deep.loc[dates[-1], "IMPACT"] < last["IMPACT"]


def test_trend_quality_r2() -> None:
    dates = _dates(60)
    t = np.arange(60, dtype=float)
    close = pd.DataFrame(
        {
            "TREND": 100.0 * np.exp(0.005 * t),
            "WOBBLE": 100.0 * np.exp(0.10 * np.sin(t / 3.0)),
        },
        index=dates,
    )
    quality = trend_quality(close, window=30)
    _assert_full_panel(quality, close)
    assert quality.loc[dates[:29]].isna().all()

    clean = quality.dropna()
    assert ((clean >= 0.0) & (clean <= 1.0)).all()
    last = quality.loc[dates[-1]]
    assert last["TREND"] == pytest.approx(1.0, abs=1e-8)
    assert last["WOBBLE"] < 0.9
    assert last["TREND"] > last["WOBBLE"]


def test_factor_validation_errors() -> None:
    dates = _dates(3)
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
    volume = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    with pytest.raises(ValueError, match="window"):
        volatility(returns, window=1)
    with pytest.raises(ValueError, match="window"):
        short_term_reversal(returns, window=0)
    with pytest.raises(ValueError, match="window"):
        amihud(returns, volume, window=0)
    with pytest.raises(ValueError, match="window"):
        trend_quality(pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates), window=1)
    with pytest.raises(ValueError, match="non-negative"):
        amihud(returns, pd.DataFrame({"A": [-1.0, 1.0, 1.0]}, index=dates))
