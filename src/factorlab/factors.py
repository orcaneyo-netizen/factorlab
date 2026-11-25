"""
factorlab.factors — Cross-sectional factor library.

Every factor takes wide frames (dates x assets) and returns a long
``pd.Series`` with MultiIndex ``(date, asset)`` covering the full panel
product, including NaN during the warm-up window.  Values at date ``t`` only
use information available at the close of ``t`` (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import to_long


def _check_wide(df: pd.DataFrame, name: str) -> None:
    """Validate a wide price/return frame."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if df.empty:
        raise ValueError(f"{name} must not be empty")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted ascending")


def _positive_wide(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate a strictly positive wide frame (prices, market cap)."""
    _check_wide(df, name)
    if (df.to_numpy(dtype="float64") <= 0).any():
        raise ValueError(f"{name} must be strictly positive")
    return df.astype("float64")


def momentum(
    close_wide: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
) -> pd.Series:
    """
    Momentum: cumulative return over ``lookback`` days, skipping the most
    recent ``skip`` days.

    Formula::

        mom_t = prod_{j=skip+1}^{skip+lookback} (1 + r_{t-j}) - 1

    i.e. the compounded return from ``t - skip - lookback`` to ``t - skip``.
    Skipping the last month avoids contaminating momentum with short-term
    reversal.  Implemented as a rolling sum of log returns on
    ``pct_change().shift(skip)``.

    Args:
        close_wide: Wide close prices (dates x assets).
        lookback: Length of the formation window in trading days (>= 1).
        skip: Number of most recent days to skip (>= 0).

    Returns:
        Long Series named ``momentum``; NaN for the first
        ``lookback + skip`` rows (the first ``pct_change`` is NaN).
    """
    close_wide = _positive_wide(close_wide, "close_wide")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if skip < 0:
        raise ValueError("skip must be >= 0")

    returns = close_wide.pct_change(fill_method=None)
    log_ret = np.log1p(returns).replace([np.inf, -np.inf], np.nan)
    mom = log_ret.shift(skip).rolling(lookback, min_periods=lookback).sum()
    return to_long(np.expm1(mom), "momentum")


def short_term_reversal(
    returns_wide: pd.DataFrame,
    window: int = 5,
) -> pd.Series:
    """
    Short-term reversal: negative of the recent cumulative return.

    Formula::

        rev_t = - sum_{j=0}^{window-1} r_{t-j}

    Assets that rose over the last week tend to underperform next period, so
    the factor is negated: a higher value means a more negative recent return.

    Args:
        returns_wide: Wide simple returns (dates x assets).
        window: Lookback window in days (>= 1).

    Returns:
        Long Series named ``short_term_reversal``; NaN before warm-up.
    """
    _check_wide(returns_wide, "returns_wide")
    if window < 1:
        raise ValueError("window must be >= 1")

    rev = -returns_wide.astype("float64").rolling(
        window, min_periods=window
    ).sum()
    return to_long(rev, "short_term_reversal")


def volatility(
    returns_wide: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """
    Low-volatility factor: negative rolling standard deviation of returns.

    Formula::

        vol_t   = std(r_{t-window+1}, ..., r_t, ddof=1)
        factor_t = -vol_t

    The negation encodes the low-volatility anomaly: lower risk is expected to
    earn higher risk-adjusted returns, so the *factor* increases as volatility
    falls.  A noisier asset therefore gets a **lower** factor value.

    Args:
        returns_wide: Wide simple returns (dates x assets).
        window: Rolling window in days (>= 2).

    Returns:
        Long Series named ``volatility``; NaN before warm-up.
    """
    _check_wide(returns_wide, "returns_wide")
    if window < 2:
        raise ValueError("window must be >= 2 for a standard deviation")

    vol = returns_wide.astype("float64").rolling(
        window, min_periods=window
    ).std(ddof=1)
    return to_long(-vol, "volatility")


def liquidity(
    volume_wide: pd.DataFrame,
    market_cap_wide: pd.DataFrame,
) -> pd.Series:
    """
    Liquidity: cross-sectional z-score of daily turnover.

    Formula::

        turnover_t = volume_t / market_cap_t
        factor_t   = (turnover_t - mean_cs) / std_cs(turnover_t, ddof=0)

    A high value means unusually high turnover (a more liquid / more actively
    traded name) relative to the rest of the cross-section on that date.

    Args:
        volume_wide: Wide share volume (dates x assets).
        market_cap_wide: Wide market capitalization (dates x assets).

    Returns:
        Long Series named ``liquidity``; NaN where volume or cap is missing.
    """
    _check_wide(volume_wide, "volume_wide")
    market_cap_wide = _positive_wide(market_cap_wide, "market_cap_wide")
    volume_wide = volume_wide.astype("float64").reindex_like(market_cap_wide)
    if (volume_wide.to_numpy() < 0).any():
        raise ValueError("volume_wide must be non-negative")

    turnover = volume_wide / market_cap_wide
    mean = turnover.mean(axis=1)
    std = turnover.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = turnover.sub(mean, axis=0).div(std, axis=0)
    return to_long(z, "liquidity")


def amihud(
    returns_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """
    Amihud (2002) illiquidity factor.

    Formula::

        illiq_t = mean_{j=0}^{window-1} |r_{t-j}| / dollar_volume_{t-j}

    where ``volume`` is share volume (pass a dollar-volume frame if one is
    available, since the panel signature has no separate price input).
    Higher values mean a larger price impact per unit traded, so
    the factor is a positive illiquidity measure (the illiquidity premium
    predicts higher returns for illiquid names).

    Args:
        returns_wide: Wide simple returns (dates x assets).
        volume_wide: Wide share volume (dates x assets).
        window: Rolling window in days (>= 1).

    Returns:
        Long Series named ``amihud``; NaN before warm-up and where volume is 0.
    """
    _check_wide(returns_wide, "returns_wide")
    _check_wide(volume_wide, "volume_wide")
    if window < 1:
        raise ValueError("window must be >= 1")

    volume_wide = volume_wide.astype("float64").reindex_like(returns_wide)
    if (volume_wide.to_numpy() < 0).any():
        raise ValueError("volume_wide must be non-negative")

    daily_illiq = returns_wide.astype("float64").abs() / volume_wide.replace(
        0.0, np.nan
    )
    illiq = daily_illiq.rolling(window, min_periods=window).mean()
    return to_long(illiq, "amihud")


def trend_quality(
    close_wide: pd.DataFrame,
    window: int = 126,
) -> pd.Series:
    """
    Trend quality: R^2 of log price regressed on a linear time trend.

    For each rolling window of length ``window``::

        log P_j = a + b * j + e_j,   j = 0, ..., window-1
        R^2 = 1 - SS_res / SS_tot

    A high R^2 means the price path is a clean (log-)linear trend; a low R^2
    means it is noisy or directionless.  R^2 is in ``[0, 1]`` (NaN if the
    window has zero variance or missing values).

    Args:
        close_wide: Wide close prices (dates x assets).
        window: Rolling regression window in days (>= 2).

    Returns:
        Long Series named ``trend_quality``; NaN before warm-up.
    """
    close_wide = _positive_wide(close_wide, "close_wide")
    if window < 2:
        raise ValueError("window must be >= 2")

    log_price = np.log(close_wide)

    x = np.arange(window, dtype="float64")
    x = x - x.mean()
    x_ss = float(np.dot(x, x))

    def _r2(window_values: np.ndarray) -> float:
        y = window_values - window_values.mean()
        ss_tot = float(np.dot(y, y))
        if not np.isfinite(ss_tot) or ss_tot <= 0.0:
            return np.nan
        beta = float(np.dot(x, y)) / x_ss
        resid = y - beta * x
        return 1.0 - float(np.dot(resid, resid)) / ss_tot

    r2 = log_price.rolling(window, min_periods=window).apply(_r2, raw=True)
    return to_long(r2, "trend_quality")
