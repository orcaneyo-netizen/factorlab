"""
factorlab.preprocess — Cross-sectional preprocessing utilities.

Every function takes a long ``pd.Series`` indexed by ``(date, asset)`` and
operates per date (cross-section), except :func:`fillna_cross_section`, which
explicitly combines a time-series fill with a cross-sectional fallback.

All functions preserve the input index exactly and propagate NaN rather than
silently filling it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_PPF_EPS = 1e-12


def _check_long_series(s: pd.Series, name: str = "s") -> None:
    """Validate the long panel Series contract."""
    if not isinstance(s, pd.Series):
        raise ValueError(f"{name} must be a pandas Series")
    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError(f"{name} must have a MultiIndex (date, asset)")
    if list(s.index.names) != ["date", "asset"]:
        raise ValueError(f"{name} MultiIndex names must be ['date', 'asset']")
    if not s.index.is_unique:
        raise ValueError(f"{name} index must be unique")


def winsorize(
    s: pd.Series,
    limits: tuple[float, float] = (0.01, 0.99),
) -> pd.Series:
    """
    Cross-sectional winsorization at specified quantiles.

    For each date, values below the ``limits[0]`` quantile are set to that
    quantile and values above the ``limits[1]`` quantile are set to it::

        x' = clip(x, Q_{lower}(x | date), Q_{upper}(x | date))

    Quantiles ignore NaN; NaN stays NaN.

    Args:
        s: Series with MultiIndex ``(date, asset)``.
        limits: ``(lower_quantile, upper_quantile)`` with ``0 <= lower < upper <= 1``.

    Returns:
        Winsorized Series with the same index.

    Raises:
        ValueError: On a malformed index or invalid limits.
    """
    _check_long_series(s)
    lower_q, upper_q = limits
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("limits must satisfy 0 <= lower < upper <= 1")

    grouped = s.groupby(level="date")
    lower = grouped.transform(lambda x: x.quantile(lower_q))
    upper = grouped.transform(lambda x: x.quantile(upper_q))
    # clip with NaN bounds is a no-op for all-NaN groups.
    return s.clip(lower, upper).reindex(s.index)


def zscore(s: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score standardization.

    ``z = (x - mean(x)) / std(x, ddof=0)`` per date.  Dates with zero or NaN
    standard deviation produce NaN (never a divide-by-zero).

    Args:
        s: Series with MultiIndex ``(date, asset)``.

    Returns:
        Z-scored Series with the same index.
    """
    _check_long_series(s)
    grouped = s.groupby(level="date")
    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0))
    return ((s - mean) / std.replace(0.0, np.nan)).reindex(s.index)


def rank_normal(s: pd.Series) -> pd.Series:
    """
    Cross-sectional rank normalization to a standard normal.

    For each date with ``n >= 2`` observations::

        u_i = rank(x_i) / (n + 1)          # average ranks, in (0, 1)
        z_i = Phi^{-1}(clip(u_i, eps, 1 - eps))

    The ``1 / (n + 1)`` convention (Blom-style plotting positions) keeps
    ``u`` strictly inside ``(0, 1)``; the explicit clip to ``1e-12`` guards
    against infinities for pathological inputs.  NaN stays NaN; dates with
    fewer than two observations return NaN.

    Args:
        s: Series with MultiIndex ``(date, asset)``.

    Returns:
        Rank-normal Series with the same index, approximately N(0, 1).
    """
    _check_long_series(s)
    grouped = s.groupby(level="date")
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    uniform = ranks / (counts + 1.0)
    clipped = uniform.clip(lower=_PPF_EPS, upper=1.0 - _PPF_EPS)
    values = stats.norm.ppf(clipped.to_numpy(dtype="float64"))
    out = pd.Series(values, index=s.index, name=s.name)
    return out.where(counts >= 2)


def demean(s: pd.Series) -> pd.Series:
    """
    Cross-sectional demeaning.

    ``x' = x - mean(x)`` per date; NaN is preserved.

    Args:
        s: Series with MultiIndex ``(date, asset)``.

    Returns:
        Demeaned Series with the same index.
    """
    _check_long_series(s)
    mean = s.groupby(level="date").transform("mean")
    return (s - mean).reindex(s.index)


def neutralize(
    s: pd.Series,
    exposures: pd.DataFrame,
    add_const: bool = True,
) -> pd.Series:
    """
    Cross-sectional neutralization via OLS residuals.

    For each date, regress the factor on the exposure columns (plus an
    optional intercept) and keep the residual::

        y_t = X_t beta_t + e_t
        beta_t = argmin_b ||y_t - X_t b||_2      (numpy.linalg.lstsq)
        factor_neutral_t = e_t

    By the normal equations ``X_t' e_t = 0``, so the residuals are orthogonal
    to every exposure.  With ``add_const=True`` they also sum to zero, hence
    their Pearson correlation with every exposure is zero (to machine
    precision).  Rows with a missing factor or exposure are excluded from the
    regression and returned as NaN; dates with ``n <= k`` usable rows return
    all NaN.

    Args:
        s: Dependent variable, MultiIndex ``(date, asset)``.
        exposures: Exposure columns, same MultiIndex as ``s``.
        add_const: Whether to append an intercept column.

    Returns:
        Residual Series with the same index as ``s``.

    Raises:
        ValueError: If indices are malformed or do not match exactly.
    """
    _check_long_series(s)
    if not isinstance(exposures, pd.DataFrame):
        raise ValueError("exposures must be a pandas DataFrame")
    if not isinstance(exposures.index, pd.MultiIndex):
        raise ValueError("exposures must have a MultiIndex (date, asset)")
    if list(exposures.index.names) != ["date", "asset"]:
        raise ValueError("exposures MultiIndex names must be ['date', 'asset']")
    if exposures.shape[1] == 0:
        raise ValueError("exposures must contain at least one column")
    if not s.index.equals(exposures.index):
        raise ValueError("s and exposures must have identical index")

    n_expo = exposures.shape[1]
    n_params = n_expo + (1 if add_const else 0)
    result = pd.Series(np.nan, index=s.index, dtype="float64", name=s.name)

    for date, y_group in s.groupby(level="date", sort=True):
        x_group = exposures.xs(date, level="date")
        mask = y_group.notna() & x_group.notna().all(axis=1)
        if int(mask.sum()) <= n_params:
            continue
        y = y_group.to_numpy(dtype="float64")[mask.to_numpy()]
        x = x_group.to_numpy(dtype="float64")[mask.to_numpy()]
        if add_const:
            x = np.column_stack([np.ones(len(y)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ beta
        result.loc[y_group.index[mask.to_numpy()]] = residuals

    return result.reindex(s.index)


def fillna_cross_section(
    s: pd.Series,
    method: str = "median",
    max_gap: int = 5,
) -> pd.Series:
    """
    Fill missing values with a bounded time-series fill, then optionally a
    cross-sectional fallback.

    * ``method='ffill'``: forward-fill within each asset, at most ``max_gap``
      consecutive periods.
    * ``method='bfill'``: backward-fill with the same limit.
    * ``method='median'``: forward-fill with the limit, then fill any remaining
      NaN with that date's cross-sectional median.

    Forward-filling more than ``max_gap`` periods is refused: stale prices are
    a classic source of look-ahead-free but economically fake signals.

    Args:
        s: Series with MultiIndex ``(date, asset)``.
        method: ``'ffill'``, ``'bfill'`` or ``'median'``.
        max_gap: Maximum number of consecutive periods to fill (>= 0).

    Returns:
        Filled Series with the same index.

    Raises:
        ValueError: On an unknown method or a negative ``max_gap``.
    """
    _check_long_series(s)
    if method not in ("ffill", "bfill", "median"):
        raise ValueError("method must be 'ffill', 'bfill', or 'median'")
    if not isinstance(max_gap, (int, np.integer)) or max_gap < 0:
        raise ValueError("max_gap must be a non-negative integer")

    if max_gap == 0:
        return s.copy()

    wide = s.unstack(level="asset")

    if method == "ffill":
        filled = wide.ffill(limit=int(max_gap))
    elif method == "bfill":
        filled = wide.bfill(limit=int(max_gap))
    else:
        filled = wide.ffill(limit=int(max_gap))
        cs_median = filled.median(axis=1)
        filled = filled.mask(filled.isna(), cs_median, axis=0)

    # stack() drops NaN rows in pandas 3; reindex restores them.
    return filled.stack().reindex(s.index).rename(s.name)
