"""Cross-sectional preprocessing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from factorlab.panel import to_long
from factorlab.preprocess import (
    demean,
    fillna_cross_section,
    neutralize,
    rank_normal,
    winsorize,
    zscore,
)


def make_long(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    assets: list[str],
    name: str = "factor",
) -> pd.Series:
    """Build a long Series from a ``(dates, assets)`` matrix."""
    return to_long(pd.DataFrame(values, index=dates, columns=assets), name=name)


def _one_date(values: list[float], assets: list[str] | None = None) -> pd.Series:
    dates = pd.date_range("2022-01-03", periods=1, freq="B")
    if assets is None:
        assets = [f"A{i:02d}" for i in range(len(values))]
    return make_long(np.asarray(values, dtype=float)[None, :], dates, assets)


def test_winsorize_clips_at_exact_quantiles() -> None:
    s = _one_date(list(range(10)))
    out = winsorize(s, limits=(0.1, 0.9))
    assert out.min() == pytest.approx(0.9)
    assert out.max() == pytest.approx(8.1)
    # The middle values are untouched.
    np.testing.assert_allclose(out.iloc[1:9], np.arange(1, 9), atol=1e-12)


def test_winsorize_per_date_and_nan() -> None:
    dates = pd.date_range("2022-01-03", periods=2, freq="B")
    values = np.array([[0, 1, 2, 3, 100], [0, 10, 20, 30, 40]], dtype=float)
    values[0, 4] = np.nan
    s = make_long(values, dates, [f"A{i}" for i in range(5)])
    out = winsorize(s, limits=(0.2, 0.8))
    assert np.isnan(out.loc[(dates[0], "A4")])
    assert out.loc[(dates[0], "A0")] >= 0.4  # 20% quantile of [0,1,2,3]
    assert out.loc[(dates[1], "A4")] < 40.0


def test_winsorize_validation() -> None:
    s = _one_date([1.0, 2.0])
    with pytest.raises(ValueError, match="limits"):
        winsorize(s, limits=(0.9, 0.1))
    with pytest.raises(ValueError, match="MultiIndex"):
        winsorize(pd.Series([1.0, 2.0]))


def test_zscore_mean_zero_std_one() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=8, freq="B")
    values = rng.normal(size=(8, 20)) * 5 + 3
    s = make_long(values, dates, [f"A{i}" for i in range(20)])
    out = zscore(s)
    for date in dates:
        row = out.loc[date]
        assert row.mean() == pytest.approx(0.0, abs=1e-12)
        assert row.std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_zscore_zero_variance_is_nan() -> None:
    s = _one_date([2.0, 2.0, 2.0])
    out = zscore(s)
    assert out.isna().all()


def test_rank_normal_is_standard_normal() -> None:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2022-01-03", periods=5, freq="B")
    values = rng.normal(size=(5, 200))
    values[0, 0] = np.nan
    s = make_long(values, dates, [f"A{i:03d}" for i in range(200)])
    out = rank_normal(s)

    assert np.isnan(out.loc[(dates[0], "A000")])
    assert np.isfinite(out.dropna()).all()

    clean = out.dropna()
    assert clean.mean() == pytest.approx(0.0, abs=1e-10)
    assert clean.std(ddof=0) == pytest.approx(1.0, abs=0.03)

    # Exact mapping for a clean cross-section.
    row = out.loc[dates[1]]
    expected = stats.norm.ppf(np.arange(1, 201) / 201.0)
    np.testing.assert_allclose(np.sort(row.to_numpy()), expected, atol=1e-12)


def test_rank_normal_small_cross_section() -> None:
    s = _one_date([1.0, np.nan])
    assert rank_normal(s).isna().all()


def test_demean() -> None:
    s = _one_date([1.0, 2.0, 6.0])
    out = demean(s)
    assert out.mean() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(out.to_numpy(), [-2.0, -1.0, 3.0])


def _neutralize_panel(
    n_dates: int = 20, n_assets: int = 60, seed: int = 11
) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n_dates, freq="B")
    assets = [f"A{i:03d}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

    beta = rng.normal(size=(n_dates, n_assets))
    size = rng.normal(size=(n_dates, n_assets))
    value = rng.normal(size=(n_dates, n_assets))
    noise = rng.normal(scale=0.1, size=(n_dates, n_assets))
    y = 1.0 + 2.0 * beta - 1.5 * size + 0.5 * value + noise

    exposures = pd.DataFrame(
        {"beta": beta.ravel(), "size": size.ravel(), "value": value.ravel()},
        index=index,
    )
    factor = pd.Series(y.ravel(), index=index, name="factor")
    return factor, exposures


def test_neutralize_residuals_orthogonal_to_exposures() -> None:
    factor, exposures = _neutralize_panel()
    residuals = neutralize(factor, exposures, add_const=True)

    assert residuals.index.equals(factor.index)
    assert residuals.notna().all()
    assert residuals.groupby(level="date").mean().abs().max() < 1e-10

    res_wide = residuals.unstack(level="asset")
    for date in res_wide.index:
        r = res_wide.loc[date].to_numpy()
        for col in exposures.columns:
            x = exposures.loc[date, col].to_numpy()
            corr = np.corrcoef(r, x)[0, 1]
            assert abs(corr) < 1e-8, f"date {date} exposure {col}: corr={corr}"


def test_neutralize_without_intercept() -> None:
    factor, exposures = _neutralize_panel(n_dates=5, n_assets=40, seed=12)
    residuals = neutralize(factor, exposures, add_const=False)
    res_wide = residuals.unstack(level="asset")
    for date in res_wide.index:
        r = res_wide.loc[date].to_numpy()
        x = exposures.loc[date].to_numpy()
        # OLS orthogonality: X' r = 0.  Without an intercept the residuals do
        # not sum to zero, so Pearson correlation need not vanish; the inner
        # product is the correct invariant.
        assert np.abs(x.T @ r).max() < 1e-10 * max(1.0, np.abs(r).max())
    # With an intercept the residuals also have zero mean, hence zero corr.
    with_intercept = neutralize(factor, exposures, add_const=True)
    assert with_intercept.groupby(level="date").mean().abs().max() < 1e-10


def test_neutralize_handles_missing_rows() -> None:
    factor, exposures = _neutralize_panel(n_dates=2, n_assets=30, seed=13)
    factor.iloc[0] = np.nan
    out = neutralize(factor, exposures)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:30].notna().all()
    assert out.loc[factor.index[-1][0]].notna().all()


def test_neutralize_validation() -> None:
    factor, exposures = _neutralize_panel(n_dates=2, n_assets=10, seed=14)
    with pytest.raises(ValueError, match="identical index"):
        neutralize(factor.iloc[:-1], exposures)
    with pytest.raises(ValueError, match="at least one column"):
        neutralize(factor, exposures.iloc[:, :0])


def test_fillna_respects_max_gap() -> None:
    dates = pd.date_range("2022-01-03", periods=5, freq="B")
    assets = ["A", "B", "C"]
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [np.nan, 2.0, 3.0],
            [np.nan, 2.0, 3.0],
            [np.nan, 2.0, 3.0],
            [5.0, 2.0, 3.0],
        ]
    )
    s = make_long(values, dates, assets)

    ffill = fillna_cross_section(s, method="ffill", max_gap=2)
    assert ffill.loc[(dates[1], "A")] == 1.0
    assert ffill.loc[(dates[2], "A")] == 1.0
    assert np.isnan(ffill.loc[(dates[3], "A")])  # third gap is refused
    assert ffill.loc[(dates[4], "A")] == 5.0

    median = fillna_cross_section(s, method="median", max_gap=2)
    assert median.loc[(dates[3], "A")] == pytest.approx(2.5)
    assert median.notna().all()

    bfill = fillna_cross_section(s, method="bfill", max_gap=1)
    assert bfill.loc[(dates[3], "A")] == 5.0
    assert np.isnan(bfill.loc[(dates[1], "A")])


def test_fillna_zero_gap_and_validation() -> None:
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    s = make_long(np.array([[1.0], [np.nan], [3.0]]), dates, ["A"])
    pd.testing.assert_series_equal(
        fillna_cross_section(s, max_gap=0), s, check_names=True
    )
    with pytest.raises(ValueError, match="max_gap"):
        fillna_cross_section(s, max_gap=-1)
    with pytest.raises(ValueError, match="method"):
        fillna_cross_section(s, method="interpolate")
