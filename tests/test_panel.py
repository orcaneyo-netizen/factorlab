"""Panel contract, determinism and wide/long round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlab.panel import (
    PANEL_COLUMNS,
    PanelData,
    from_wide,
    synthetic_panel,
    to_long,
    to_wide,
    validate_panel,
)


def test_shape_index_and_dtypes(long_panel: pd.DataFrame) -> None:
    assert long_panel.shape == (12 * 160, len(PANEL_COLUMNS))
    assert isinstance(long_panel.index, pd.MultiIndex)
    assert list(long_panel.index.names) == ["date", "asset"]
    assert list(long_panel.columns) == list(PANEL_COLUMNS)
    for col in PANEL_COLUMNS:
        assert pd.api.types.is_float_dtype(long_panel[col])
    assert long_panel.index.is_monotonic_increasing
    assert long_panel.index.is_unique


def test_determinism() -> None:
    a = synthetic_panel(n_assets=6, n_days=40, seed=123)
    b = synthetic_panel(n_assets=6, n_days=40, seed=123)
    c = synthetic_panel(n_assets=6, n_days=40, seed=124)
    pd.testing.assert_frame_equal(a, b)
    assert not a["returns"].equals(c["returns"])


def test_cross_sectional_variance_positive(long_panel: pd.DataFrame) -> None:
    cs_var = long_panel["returns"].groupby(level="date").var(ddof=0)
    assert (cs_var > 0).all()
    assert cs_var.notna().all()


def test_ohlc_and_return_consistency(long_panel: pd.DataFrame) -> None:
    assert (long_panel["high"] >= long_panel["low"]).all()
    assert (long_panel["high"] >= long_panel["close"]).all()
    assert (long_panel["low"] <= long_panel["close"]).all()
    assert (long_panel["high"] >= long_panel["open"]).all()
    assert (long_panel["low"] <= long_panel["open"]).all()

    close = to_wide(long_panel, "close")
    returns = to_wide(long_panel, "returns")
    implied = close.pct_change(fill_method=None)
    np.testing.assert_allclose(
        implied.iloc[1:].to_numpy(), returns.iloc[1:].to_numpy(), atol=1e-12
    )


def test_validate_panel_rejects_bad_frames(long_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        validate_panel(long_panel.reset_index())

    wrong_names = long_panel.copy()
    wrong_names.index = wrong_names.index.set_names(["day", "asset"])
    with pytest.raises(ValueError, match="names"):
        validate_panel(wrong_names)

    missing = long_panel.drop(columns=["volume"])
    with pytest.raises(ValueError, match="Columns"):
        validate_panel(missing)

    negative = long_panel.copy()
    negative.iloc[0, negative.columns.get_loc("close")] = -1.0
    with pytest.raises(ValueError, match="positive"):
        validate_panel(negative)

    flat = long_panel.copy()
    first_date = flat.index.get_level_values("date")[0]
    flat.loc[first_date, "returns"] = 0.0
    with pytest.raises(ValueError, match="variance"):
        validate_panel(flat)

    duplicated = pd.concat([long_panel, long_panel.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError, match="unique"):
        validate_panel(duplicated)


def test_to_wide_from_wide_roundtrip(long_panel: pd.DataFrame) -> None:
    close = to_wide(long_panel, "close")
    assert close.shape == (160, 12)
    assert list(close.columns) == sorted(close.columns)

    with_nan = close.copy()
    with_nan.iloc[3, 2] = np.nan
    long = from_wide(with_nan, "close", close.index, close.columns)
    assert isinstance(long.index, pd.MultiIndex)
    assert long.index.equals(long_panel.index)
    assert np.isnan(long.iloc[3 * len(close.columns) + 2])
    pd.testing.assert_series_equal(
        long.sort_index(), with_nan.stack().rename("close").sort_index()
    )


def test_to_long_keeps_nan_cells() -> None:
    dates = pd.date_range("2022-01-03", periods=3, freq="B")
    assets = ["A", "B"]
    wide = pd.DataFrame([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]], index=dates, columns=assets)
    long = to_long(wide, "x")
    assert len(long) == 6
    assert long.index.equals(pd.MultiIndex.from_product([dates, assets], names=["date", "asset"]))
    assert long.isna().sum() == 2
    np.testing.assert_allclose(long.dropna().to_numpy(), [1.0, 2.0, 3.0, 4.0])


def test_panel_data_accessors(panel: PanelData) -> None:
    assert len(panel.dates) == 160
    assert len(panel.assets) == 12
    for field in PANEL_COLUMNS:
        wide = panel.wide(field)
        assert wide.shape == (160, 12)
    assert panel.close.equals(panel.wide("close"))
    assert panel.returns.equals(panel.wide("returns"))


def test_synthetic_panel_input_validation() -> None:
    with pytest.raises(ValueError, match="n_assets"):
        synthetic_panel(n_assets=1, n_days=10)
    with pytest.raises(ValueError, match="n_days"):
        synthetic_panel(n_assets=3, n_days=1)
