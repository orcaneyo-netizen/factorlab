"""Shared fixtures for the factorlab test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlab.panel import PanelData, synthetic_panel


@pytest.fixture(scope="session")
def long_panel() -> pd.DataFrame:
    """A small, deterministic long panel."""
    return synthetic_panel(n_assets=12, n_days=160, seed=7)


@pytest.fixture(scope="session")
def panel(long_panel: pd.DataFrame) -> PanelData:
    """Validated PanelData wrapper around ``long_panel``."""
    return PanelData(long_panel)


def make_long(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    assets: list[str],
    name: str = "factor",
) -> pd.Series:
    """Build a long Series from a ``(dates, assets)`` matrix."""
    values = np.asarray(values, dtype="float64")
    if values.shape != (len(dates), len(assets)):
        raise ValueError("values shape does not match dates x assets")
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    return pd.Series(values.ravel("C"), index=index, name=name)
