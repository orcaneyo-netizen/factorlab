"""
factorlab.panel — Synthetic panel generation and panel utilities.

Panel convention: a :class:`pandas.DataFrame` with MultiIndex ``(date, asset)``
and columns ``open, high, low, close, volume, returns, market_cap``.  Long
format, sorted, one row per asset and trading day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PANEL_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "market_cap",
)

_INDEX_NAMES = ["date", "asset"]


def synthetic_panel(
    n_assets: int = 40,
    n_days: int = 750,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic equity panel.

    Model
    -----
    * Common market factor: ``r_m,t ~ N(mu_m, sigma_m^2)``.
    * Per-asset betas: ``beta_i ~ Uniform(0.5, 1.5)``.
    * Idiosyncratic AR(1) noise:
      ``eps_i,t = rho * eps_i,t-1 + eta_i,t``, ``eta ~ N(0, sigma_eta^2)``.
    * Asset returns: ``r_i,t = beta_i * r_m,t + eps_i,t``.
    * Prices: ``close_i,t = 100 * prod_{s<=t} (1 + r_i,s)`` so that
      ``close.pct_change()`` reproduces ``returns`` exactly.
    * Market cap: lognormal start, evolves with ``log1p(returns)`` plus noise.
    * Volume: ``base * (1 + 10 * |r|) * lognormal noise`` — correlated with
      absolute returns, as in real markets.

    Args:
        n_assets: Number of assets (>= 2).
        n_days: Number of trading days (>= 2).
        seed: Seed for :class:`numpy.random.default_rng` (deterministic).

    Returns:
        DataFrame with MultiIndex ``(date, asset)`` and the panel columns.

    Raises:
        ValueError: If ``n_assets`` or ``n_days`` is smaller than 2.
    """
    if n_assets < 2:
        raise ValueError("n_assets must be >= 2 for a cross-section")
    if n_days < 2:
        raise ValueError("n_days must be >= 2")

    rng = np.random.default_rng(seed)

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Common market factor (~5% annual return, ~16% annual vol).
    mu_m = 0.0002
    sigma_m = 0.01
    r_m = rng.normal(mu_m, sigma_m, n_days)

    # Per-asset betas.
    betas = rng.uniform(0.5, 1.5, n_assets)

    # Idiosyncratic AR(1) noise, started from its stationary distribution.
    rho = 0.1
    sigma_eta = 0.015
    eps = np.zeros((n_days, n_assets))
    eps[0] = rng.normal(0.0, sigma_eta / np.sqrt(1.0 - rho**2), n_assets)
    for t in range(1, n_days):
        eps[t] = rho * eps[t - 1] + rng.normal(0.0, sigma_eta, n_assets)

    returns = betas * r_m[:, None] + eps  # (n_days, n_assets)

    # Prices. Using log1p keeps pct_change(close) == returns to machine eps.
    log_prices = np.log(100.0) + np.cumsum(np.log1p(returns), axis=0)
    close = np.exp(log_prices)

    # OHLC: intraday noise around the close, open gaps from the previous close.
    intraday_vol = 0.005
    high = close * np.exp(rng.normal(0.0, intraday_vol, (n_days, n_assets)))
    low = close * np.exp(rng.normal(0.0, intraday_vol, (n_days, n_assets)))
    open_ = np.roll(close, 1, axis=0)
    open_[0] = 100.0
    open_ = open_ * np.exp(rng.normal(0.0, 0.002, (n_days, n_assets)))

    # Enforce the OHLC ordering high >= max(open, close) >= min(open, close) >= low.
    high = np.maximum.reduce([high, close, open_])
    low = np.minimum.reduce([low, close, open_])

    # Market cap: lognormal start (~$70B median), evolves with returns.
    log_mcap = np.zeros((n_days, n_assets))
    log_mcap[0] = rng.normal(25.0, 1.5, n_assets)
    mcap_noise = rng.normal(0.0, 0.001, (n_days, n_assets))
    for t in range(1, n_days):
        log_mcap[t] = log_mcap[t - 1] + np.log1p(returns[t]) + mcap_noise[t]
    market_cap = np.exp(log_mcap)

    # Volume: correlated with |returns| plus lognormal noise.
    base_vol = 1e6
    volume = base_vol * (1.0 + 10.0 * np.abs(returns)) * np.exp(
        rng.normal(0.0, 0.3, (n_days, n_assets))
    )

    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product([dates, assets], names=_INDEX_NAMES)

    df = pd.DataFrame(
        {
            "open": open_.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": close.ravel(),
            "volume": volume.ravel(),
            "returns": returns.ravel(),
            "market_cap": market_cap.ravel(),
        },
        index=index,
    )
    return df.sort_index()


def validate_panel(df: pd.DataFrame) -> None:
    """
    Validate that ``df`` conforms to the panel contract.

    Checks: MultiIndex named ``(date, asset)``, unique and sorted, exactly the
    seven panel columns, numeric dtypes, positive prices, non-negative volume,
    ``high >= low``, and strictly positive cross-sectional return variance on
    every date (a degenerate cross-section cannot be ranked).

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If the panel structure or values are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Panel must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Panel is empty")
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Panel must have a MultiIndex (date, asset)")
    if list(df.index.names) != _INDEX_NAMES:
        raise ValueError("MultiIndex names must be ['date', 'asset']")
    if not df.index.is_unique:
        raise ValueError("Panel index must be unique")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Panel must be sorted by (date, asset)")

    missing = set(PANEL_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(PANEL_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"Columns must be exactly {sorted(PANEL_COLUMNS)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    for col in PANEL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")

    prices = df[["open", "high", "low", "close"]]
    if (prices <= 0).to_numpy().any():
        raise ValueError("Prices must be strictly positive")
    if (df["volume"] < 0).to_numpy().any():
        raise ValueError("Volume must be non-negative")
    if (df["high"] < df["low"]).to_numpy().any():
        raise ValueError("high must be >= low on every row")
    if (df["market_cap"] <= 0).to_numpy().any():
        raise ValueError("market_cap must be strictly positive")

    cs_var = df["returns"].groupby(level="date").var(ddof=0)
    if cs_var.isna().to_numpy().any() or (cs_var <= 0).to_numpy().any():
        bad = cs_var.index[(cs_var.isna()) | (cs_var <= 0)]
        raise ValueError(
            f"Zero cross-sectional variance of returns on {list(bad[:5])}"
        )


def to_wide(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Pivot one panel column to wide format (dates x assets).

    Args:
        df: Long-format panel.
        field: Column name to pivot.

    Returns:
        Wide DataFrame with dates as index and assets as columns.

    Raises:
        ValueError: If ``field`` is not a panel column.
    """
    if field not in df.columns:
        raise ValueError(f"Field '{field}' not in panel columns")
    return df[field].unstack(level="asset")


def to_long(
    wide: pd.DataFrame,
    name: str | None = None,
    dates: pd.Index | None = None,
    assets: pd.Index | None = None,
) -> pd.Series:
    """
    Flatten a wide frame to a long Series, keeping NaN cells.

    ``DataFrame.stack`` drops missing rows in pandas 3, which would silently
    truncate factor warm-up periods.  This helper builds the full
    ``(date, asset)`` product explicitly and ravels in C order.

    Args:
        wide: Wide DataFrame (dates x assets).
        name: Name of the returned Series.
        dates: Optional date index override (defaults to ``wide.index``).
        assets: Optional asset columns override (defaults to ``wide.columns``).

    Returns:
        Series with MultiIndex ``(date, asset)``, length
        ``len(dates) * len(assets)``.
    """
    if not isinstance(wide, pd.DataFrame):
        raise ValueError("wide must be a pandas DataFrame")
    if dates is None:
        dates = wide.index
    if assets is None:
        assets = wide.columns
    dates = pd.Index(dates)
    assets = pd.Index(assets)
    aligned = wide.reindex(index=dates, columns=assets)
    index = pd.MultiIndex.from_product([dates, assets], names=_INDEX_NAMES)
    values = aligned.to_numpy(dtype="float64", copy=True).ravel("C")
    return pd.Series(values, index=index, name=name)


def from_wide(
    wide: pd.DataFrame,
    field: str,
    dates: pd.Index,
    assets: pd.Index,
) -> pd.Series:
    """
    Convert a wide frame back to a long Series aligned with a panel index.

    Args:
        wide: Wide DataFrame (dates x assets).
        field: Name for the resulting Series.
        dates: Date index from the original panel.
        assets: Asset index from the original panel.

    Returns:
        Series with MultiIndex ``(date, asset)`` in panel order.
    """
    return to_long(wide, name=field, dates=dates, assets=assets)


@dataclass
class PanelData:
    """Container for a validated long panel with wide-format accessors."""

    long: pd.DataFrame

    def __post_init__(self) -> None:
        validate_panel(self.long)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Sorted unique dates."""
        return self.long.index.get_level_values("date").unique()

    @property
    def assets(self) -> pd.Index:
        """Sorted unique assets."""
        return self.long.index.get_level_values("asset").unique()

    def wide(self, field: str) -> pd.DataFrame:
        """Wide view of one field."""
        return to_wide(self.long, field)

    @property
    def open(self) -> pd.DataFrame:
        return self.wide("open")

    @property
    def high(self) -> pd.DataFrame:
        return self.wide("high")

    @property
    def low(self) -> pd.DataFrame:
        return self.wide("low")

    @property
    def close(self) -> pd.DataFrame:
        return self.wide("close")

    @property
    def volume(self) -> pd.DataFrame:
        return self.wide("volume")

    @property
    def returns(self) -> pd.DataFrame:
        return self.wide("returns")

    @property
    def market_cap(self) -> pd.DataFrame:
        return self.wide("market_cap")
