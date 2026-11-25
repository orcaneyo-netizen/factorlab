"""Evaluation tests: forward returns, IC, quantile portfolios, turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlab.evaluate import (
    assign_quantiles,
    forward_returns,
    ic_series,
    ic_summary,
    long_short,
    quantile_returns,
    turnover,
)
from factorlab.panel import to_long


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=n, freq="B")


def _long(values: np.ndarray, dates: pd.DatetimeIndex, assets: list[str], name: str = "factor") -> pd.Series:
    return to_long(pd.DataFrame(values, index=dates, columns=assets), name=name)


def test_forward_returns_exact_sum_and_nan_tail() -> None:
    dates = _dates(6)
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]}, index=dates
    )
    fwd = forward_returns(returns, horizon=3)

    assert fwd.shape == returns.shape
    assert fwd.iloc[0, 0] == pytest.approx(0.02 + 0.03 + 0.04)
    assert fwd.iloc[1, 0] == pytest.approx(0.03 + 0.04 + 0.05)
    assert fwd.iloc[2, 0] == pytest.approx(0.04 + 0.05 + 0.06)
    assert fwd.iloc[3:, 0].isna().all()


def test_forward_returns_never_look_beyond_horizon() -> None:
    dates = _dates(10)
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(rng.normal(0, 0.01, (10, 3)), index=dates, columns=list("ABC"))
    horizon = 2
    fwd = forward_returns(returns, horizon)

    for t in range(10 - horizon):
        expected = returns.iloc[t + 1 : t + horizon + 1].sum(axis=0)
        np.testing.assert_allclose(fwd.iloc[t].to_numpy(), expected.to_numpy())

    # Perturbing row 5 must only affect windows that actually contain row 5.
    perturbed = returns.copy()
    perturbed.iloc[5] += 10.0
    fwd_perturbed = forward_returns(perturbed, horizon)
    np.testing.assert_allclose(
        fwd.iloc[:3].to_numpy(), fwd_perturbed.iloc[:3].to_numpy()
    )
    assert not np.isclose(fwd.iloc[3, 0], fwd_perturbed.iloc[3, 0])
    assert not np.isclose(fwd.iloc[4, 0], fwd_perturbed.iloc[4, 0])


def test_forward_returns_partial_window_is_nan() -> None:
    dates = _dates(4)
    returns = pd.DataFrame({"A": [0.01, np.nan, 0.03, 0.04]}, index=dates)
    fwd = forward_returns(returns, horizon=2)
    # Window starting at t=0 includes the missing r_2 -> NaN, not a partial sum.
    assert np.isnan(fwd.iloc[0, 0])
    assert fwd.iloc[1, 0] == pytest.approx(0.03 + 0.04)


def test_forward_returns_validation() -> None:
    returns = pd.DataFrame({"A": [0.01, 0.02]}, index=_dates(2))
    with pytest.raises(ValueError, match="horizon"):
        forward_returns(returns, horizon=0)
    with pytest.raises(ValueError, match="DataFrame"):
        forward_returns(returns["A"], horizon=1)


def test_ic_perfect_relationship() -> None:
    dates = _dates(1)
    assets = [f"A{i:02d}" for i in range(10)]
    factor = _long(np.arange(10.0)[None, :], dates, assets)
    fwd = pd.DataFrame(np.arange(10.0)[None, :] * 0.01, index=dates, columns=assets)

    ic = ic_series(factor, fwd)
    assert ic.iloc[0] == pytest.approx(1.0)

    ic_neg = ic_series(-factor, fwd)
    assert ic_neg.iloc[0] == pytest.approx(-1.0)


def test_ic_of_noise_is_near_zero() -> None:
    rng = np.random.default_rng(5)
    n_dates, n_assets = 200, 30
    dates = _dates(n_dates)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    factor = _long(rng.normal(size=(n_dates, n_assets)), dates, assets)
    fwd = pd.DataFrame(
        rng.normal(size=(n_dates, n_assets)), index=dates, columns=assets
    )
    ic = ic_series(factor, fwd)
    assert ic.notna().sum() == n_dates
    assert abs(ic.mean()) < 0.05
    assert ic.std(ddof=1) < 0.3


def test_ic_series_constant_cross_section_is_nan() -> None:
    dates = _dates(2)
    assets = ["A", "B", "C"]
    factor = _long(np.ones((2, 3)), dates, assets)
    fwd = pd.DataFrame(np.ones((2, 3)), index=dates, columns=assets)
    assert ic_series(factor, fwd).isna().all()


def test_ic_summary_detects_predictability() -> None:
    rng = np.random.default_rng(6)
    n_dates, n_assets = 80, 20
    dates = _dates(n_dates)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    factor_wide = rng.normal(size=(n_dates, n_assets))

    # Next-day return is strongly driven by today's factor.
    returns = np.zeros((n_dates, n_assets))
    returns[1:] = 0.05 * factor_wide[:-1] + 0.01 * rng.normal(size=(n_dates - 1, n_assets))
    returns_wide = pd.DataFrame(returns, index=dates, columns=assets)
    factor = _long(factor_wide, dates, assets)

    summary = ic_summary(factor, returns_wide, max_horizon=5, nw_lags=2)
    assert summary.mean_ic > 0.8
    assert summary.hit_rate > 0.95
    assert summary.t_stat > 5.0
    assert np.isfinite(summary.ic_ir)
    assert summary.n_obs == n_dates - 1
    assert list(summary.ic_decay.index) == [1, 2, 3, 4, 5]
    assert summary.ic_decay.iloc[0] > 0.8


def test_ic_summary_all_nan_input() -> None:
    dates = _dates(5)
    assets = [f"A{i}" for i in range(5)]
    factor = _long(np.full((5, 5), np.nan), dates, assets)
    returns = pd.DataFrame(np.zeros((5, 5)), index=dates, columns=assets)
    summary = ic_summary(factor, returns, max_horizon=3)
    assert summary.n_obs == 0
    assert np.isnan(summary.mean_ic)
    assert len(summary.ic_decay) == 3


def test_quantile_portfolios_are_balanced() -> None:
    dates = _dates(1)
    assets = [f"A{i:02d}" for i in range(10)]
    factor = _long(np.arange(10.0)[None, :], dates, assets)
    fwd = pd.DataFrame(np.arange(10.0)[None, :], index=dates, columns=assets)

    q = quantile_returns(factor, fwd, n_quantiles=5)
    assert list(q.columns) == [1, 2, 3, 4, 5]
    # Equal-count buckets: Q1 = {0,1}, Q2 = {2,3}, ...
    np.testing.assert_allclose(q.iloc[0].to_numpy(), [0.5, 2.5, 4.5, 6.5, 8.5])

    labels = assign_quantiles(factor, 5)
    counts = labels.value_counts().sort_index()
    assert list(counts.index) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert counts.to_numpy().tolist() == [2, 2, 2, 2, 2]


def test_quantile_cap_weighting_and_long_short() -> None:
    dates = _dates(2)
    assets = [f"A{i}" for i in range(6)]
    factor = _long(np.tile(np.arange(6.0), (2, 1)), dates, assets)
    fwd = pd.DataFrame(np.full((2, 6), 0.02), index=dates, columns=assets)
    caps = pd.DataFrame(np.tile(np.arange(1.0, 7.0), (2, 1)), index=dates, columns=assets)

    q = quantile_returns(factor, fwd, n_quantiles=3, weighting="cap", market_cap=caps)
    np.testing.assert_allclose(q.to_numpy(), 0.02)

    ls = long_short(q)
    np.testing.assert_allclose(ls.to_numpy(), 0.0)


def test_long_short_top_minus_bottom() -> None:
    dates = _dates(3)
    q = pd.DataFrame(
        {1: [0.01, 0.02, 0.03], 2: [0.02, 0.03, 0.04], 3: [0.05, 0.06, 0.07]},
        index=dates,
    )
    ls = long_short(q)
    np.testing.assert_allclose(ls.to_numpy(), [0.04, 0.04, 0.04])


def test_turnover_constant_and_flipping() -> None:
    dates = _dates(2)
    assets = [f"A{i}" for i in range(10)]
    same = _long(np.tile(np.arange(10.0), (2, 1)), dates, assets)
    t_same = turnover(same, n_quantiles=2)
    assert np.isnan(t_same.iloc[0])
    assert t_same.iloc[1] == pytest.approx(0.0)

    flipped_values = np.vstack([np.arange(10.0), -np.arange(10.0)])
    flipped = _long(flipped_values, dates, assets)
    t_flip = turnover(flipped, n_quantiles=2)
    assert t_flip.iloc[1] == pytest.approx(1.0)


def test_quantile_validation() -> None:
    dates = _dates(2)
    assets = ["A", "B", "C"]
    factor = _long(np.arange(6.0).reshape(2, 3), dates, assets)
    fwd = pd.DataFrame(np.arange(6.0).reshape(2, 3), index=dates, columns=assets)
    with pytest.raises(ValueError, match="weighting"):
        quantile_returns(factor, fwd, weighting="vwap")
    with pytest.raises(ValueError, match="market_cap"):
        quantile_returns(factor, fwd, weighting="cap")
    with pytest.raises(ValueError, match="n_quantiles"):
        quantile_returns(factor, fwd, n_quantiles=1)
    with pytest.raises(ValueError, match="at least 2"):
        long_short(pd.DataFrame({1: [0.1]}, index=dates))
    with pytest.raises(ValueError, match="method"):
        ic_series(factor, fwd, method="kendall")
