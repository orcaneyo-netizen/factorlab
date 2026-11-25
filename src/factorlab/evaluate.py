"""
factorlab.evaluate — Factor evaluation: IC analysis, quantile portfolios, turnover.

Definitions
-----------
* Forward return: ``fwd_t = sum_{j=1..h} r_{t+j}`` — strictly future data, no
  information beyond ``t + h``.
* IC: cross-sectional Spearman (or Pearson) correlation between the factor at
  ``t`` and the forward return starting at ``t``.
* Quantile portfolios: assets sorted by factor each date, split into
  ``n_quantiles`` equal-count buckets (Q1 = lowest factor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

_INDEX_NAMES = ["date", "asset"]


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------


def forward_returns(
    returns_wide: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Sum of simple returns over the next ``horizon`` days.

    Formula::

        fwd_t = sum_{j=1}^{horizon} r_{t+j}

    The last ``horizon`` rows are NaN because a complete forward window does
    not exist.  A partial window is *never* returned: if any of the ``horizon``
    future returns is missing, the result is NaN.

    Args:
        returns_wide: Wide simple returns (dates x assets).
        horizon: Forward horizon in trading days (>= 1).

    Returns:
        Wide DataFrame of forward returns with the same index and columns.

    Raises:
        ValueError: If ``horizon < 1`` or the input is not a DataFrame.
    """
    if not isinstance(returns_wide, pd.DataFrame):
        raise ValueError("returns_wide must be a pandas DataFrame")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be an integer >= 1")

    values = returns_wide.to_numpy(dtype="float64")
    # stack h shifted copies: shape (h, n_dates, n_assets)
    stacked = np.stack(
        [np.roll(values, -h, axis=0) for h in range(1, int(horizon) + 1)],
        axis=0,
    )
    fwd = stacked.sum(axis=0)
    # np.roll wraps around; the last `horizon` rows are invalid.
    fwd[-int(horizon) :, :] = np.nan
    # Any NaN inside the window must poison the sum (np.sum already does this,
    # but the explicit mask documents the contract).
    for h in range(1, int(horizon) + 1):
        valid = np.roll(~np.isnan(values), -h, axis=0)
        valid[-h:, :] = False
        fwd = np.where(valid, fwd, np.nan)
    return pd.DataFrame(fwd, index=returns_wide.index, columns=returns_wide.columns)


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------


def _factor_wide(factor: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """Unstack a long factor and align it to a wide reference frame."""
    if not isinstance(factor, pd.Series):
        raise ValueError("factor must be a pandas Series")
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("factor must have a MultiIndex (date, asset)")
    if list(factor.index.names) != _INDEX_NAMES:
        raise ValueError("factor MultiIndex names must be ['date', 'asset']")
    if not factor.index.is_unique:
        raise ValueError("factor index must be unique")
    wide = factor.unstack(level="asset")
    return wide.reindex(index=like.index, columns=like.columns)


def ic_series(
    factor: pd.Series,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    Cross-sectional Information Coefficient per date.

    ``IC_t = corr(factor_t, fwd_return_t)`` across assets.  Dates with fewer
    than three usable assets or a constant factor/return produce NaN.

    Args:
        factor: Long Series with MultiIndex ``(date, asset)``.
        fwd_returns: Wide forward returns (dates x assets).
        method: ``'spearman'`` (rank correlation) or ``'pearson'``.

    Returns:
        Series indexed by date, named ``ic``.

    Raises:
        ValueError: On a malformed factor, method, or forward-return frame.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")
    if not isinstance(fwd_returns, pd.DataFrame):
        raise ValueError("fwd_returns must be a pandas DataFrame")

    factor_wide = _factor_wide(factor, fwd_returns)

    ics = np.full(len(fwd_returns), np.nan, dtype="float64")
    for i, date in enumerate(fwd_returns.index):
        f_row = factor_wide.iloc[i]
        r_row = fwd_returns.iloc[i]
        mask = f_row.notna() & r_row.notna()
        if int(mask.sum()) < 3:
            continue
        f_clean = f_row[mask].to_numpy(dtype="float64")
        r_clean = r_row[mask].to_numpy(dtype="float64")
        if np.ptp(f_clean) == 0.0 or np.ptp(r_clean) == 0.0:
            continue
        if method == "spearman":
            ic, _ = stats.spearmanr(f_clean, r_clean)
        else:
            ic, _ = stats.pearsonr(f_clean, r_clean)
        ics[i] = ic

    return pd.Series(ics, index=fwd_returns.index, name="ic")


@dataclass
class ICSummary:
    """Summary statistics of an IC series."""

    mean_ic: float
    ic_ir: float  # mean(IC) / std(IC), per-period information ratio
    t_stat: float  # Newey-West HAC t-statistic for mean(IC) = 0
    hit_rate: float  # fraction of dates with IC > 0
    ic_decay: pd.Series  # mean IC at horizons 1..max_horizon
    n_obs: int

    def to_dict(self) -> dict:
        """Plain-dict view (``ic_decay`` as a dict keyed by horizon)."""
        return {
            "mean_ic": self.mean_ic,
            "ic_ir": self.ic_ir,
            "t_stat": self.t_stat,
            "hit_rate": self.hit_rate,
            "ic_decay": {int(k): float(v) for k, v in self.ic_decay.items()},
            "n_obs": self.n_obs,
        }


def _newey_west_tstat(x: np.ndarray, lags: int) -> float:
    """
    Newey-West (HAC) t-statistic for the sample mean of ``x``.

    ``Var_hat(mean) = (gamma_0 + 2 * sum_{j=1..L} (1 - j/(L+1)) gamma_j) / n``
    with ``gamma_j`` the j-th autocovariance of the demeaned series.  Returns
    NaN for fewer than two observations or non-positive variance.
    """
    x = np.asarray(x, dtype="float64")
    n = x.size
    if n < 2:
        return np.nan
    mu = float(x.mean())
    e = x - mu
    gamma0 = float(np.dot(e, e)) / n
    lags = int(max(0, min(lags, n - 1)))
    var = gamma0
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(e[j:], e[:-j])) / n
        var += 2.0 * (1.0 - j / (lags + 1.0)) * gamma_j
    var /= n
    if not np.isfinite(var) or var <= 0.0:
        return np.nan
    return mu / np.sqrt(var)


def ic_summary(
    factor: pd.Series,
    returns_wide: pd.DataFrame,
    max_horizon: int = 10,
    nw_lags: int = 5,
) -> ICSummary:
    """
    Summarize predictive power at horizon 1 and the IC decay curve.

    ``IC IR = mean(IC) / std(IC, ddof=1)``; the t-statistic uses a
    Newey-West HAC variance with ``nw_lags`` lags to account for overlapping
    forward-return windows.  The decay table recomputes the IC at horizons
    ``1..max_horizon`` (forward returns are non-overlapping by construction at
    each horizon, but adjacent horizons overlap).

    Args:
        factor: Long factor Series, MultiIndex ``(date, asset)``.
        returns_wide: Wide simple returns (dates x assets).
        max_horizon: Largest horizon in the decay table (>= 1).
        nw_lags: Newey-West lag truncation (>= 0).

    Returns:
        :class:`ICSummary`; all statistics are NaN when no IC is defined.
    """
    if not isinstance(max_horizon, (int, np.integer)) or max_horizon < 1:
        raise ValueError("max_horizon must be an integer >= 1")
    if not isinstance(nw_lags, (int, np.integer)) or nw_lags < 0:
        raise ValueError("nw_lags must be an integer >= 0")

    decay = []
    for h in range(1, int(max_horizon) + 1):
        ics_h = ic_series(factor, forward_returns(returns_wide, h))
        decay.append(float(ics_h.mean()) if ics_h.notna().any() else np.nan)
    ic_decay = pd.Series(
        decay, index=pd.RangeIndex(1, int(max_horizon) + 1), name="ic_decay"
    )

    ics = ic_series(factor, forward_returns(returns_wide, 1)).dropna()
    if ics.empty:
        return ICSummary(
            mean_ic=np.nan,
            ic_ir=np.nan,
            t_stat=np.nan,
            hit_rate=np.nan,
            ic_decay=ic_decay,
            n_obs=0,
        )

    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1)) if len(ics) > 1 else np.nan
    ic_ir = mean_ic / std_ic if np.isfinite(std_ic) and std_ic > 0 else np.nan
    t_stat = _newey_west_tstat(ics.to_numpy(), int(nw_lags))
    hit_rate = float((ics > 0).mean())

    return ICSummary(
        mean_ic=mean_ic,
        ic_ir=float(ic_ir) if np.isfinite(ic_ir) else np.nan,
        t_stat=float(t_stat) if np.isfinite(t_stat) else np.nan,
        hit_rate=hit_rate,
        ic_decay=ic_decay,
        n_obs=int(len(ics)),
    )


# ---------------------------------------------------------------------------
# Quantile portfolios
# ---------------------------------------------------------------------------


def assign_quantiles(
    factor: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """
    Assign each asset to a factor quantile (1 = lowest factor).

    Ranks use ``method='first'`` within each date so that every quantile is
    exactly populated (counts differ by at most one) even when the factor has
    ties.  Dates with fewer than ``n_quantiles`` valid assets get NaN.

    Args:
        factor: Long factor Series, MultiIndex ``(date, asset)``.
        n_quantiles: Number of buckets (>= 2).

    Returns:
        Float Series of quantile labels, same index as ``factor``.
    """
    if not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 2:
        raise ValueError("n_quantiles must be an integer >= 2")
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("factor must have a MultiIndex (date, asset)")

    wide = factor.unstack(level="asset")
    ranks = wide.rank(axis=1, method="first")
    counts = wide.notna().sum(axis=1)
    scaled = ranks.mul(float(n_quantiles)).div(counts, axis=0)
    labels = np.ceil(scaled.to_numpy(dtype="float64"))
    labels = np.clip(labels, 1.0, float(n_quantiles))
    labels[counts.to_numpy() < int(n_quantiles)] = np.nan
    labels_df = pd.DataFrame(labels, index=wide.index, columns=wide.columns)
    return labels_df.stack().reindex(factor.index).rename("quantile")


def quantile_returns(
    factor: pd.Series,
    fwd_returns: pd.DataFrame,
    n_quantiles: int = 5,
    weighting: str = "equal",
    market_cap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Average forward return of each factor quantile portfolio.

    Each date, assets with both a factor and a forward return are ranked into
    ``n_quantiles`` equal-count buckets (Q1 = lowest factor).  The portfolio
    return is either the equal-weighted mean or the cap-weighted mean::

        r_q = sum_{i in q} w_i r_i / sum_{i in q} w_i,   w_i = market_cap_i

    Dates with too few assets produce a row of NaN.

    Args:
        factor: Long factor Series, MultiIndex ``(date, asset)``.
        fwd_returns: Wide forward returns (dates x assets).
        n_quantiles: Number of buckets (>= 2).
        weighting: ``'equal'`` or ``'cap'``.
        market_cap: Wide market caps, required for ``weighting='cap'``.

    Returns:
        DataFrame indexed by date with columns ``1..n_quantiles``.

    Raises:
        ValueError: On invalid inputs or missing market caps.
    """
    if weighting not in ("equal", "cap"):
        raise ValueError("weighting must be 'equal' or 'cap'")
    if weighting == "cap" and market_cap is None:
        raise ValueError("market_cap required for cap-weighting")
    if not isinstance(fwd_returns, pd.DataFrame):
        raise ValueError("fwd_returns must be a pandas DataFrame")

    factor_wide = _factor_wide(factor, fwd_returns)
    q_labels = assign_quantiles(factor, n_quantiles).unstack(level="asset")
    q_labels = q_labels.reindex(index=fwd_returns.index, columns=fwd_returns.columns)

    fwd = fwd_returns.astype("float64")
    mask = q_labels.notna() & fwd.notna()

    if weighting == "cap":
        cap = market_cap.reindex(
            index=fwd_returns.index, columns=fwd_returns.columns
        ).astype("float64")
        mask = mask & cap.notna() & (cap > 0)
        weights = cap.where(mask, 0.0)
        weighted = (fwd.where(mask, 0.0) * weights).where(mask)
    else:
        weights = mask.astype("float64")
        weighted = fwd.where(mask)

    labels = q_labels.where(mask)
    out = pd.DataFrame(
        np.nan,
        index=fwd_returns.index,
        columns=pd.RangeIndex(1, int(n_quantiles) + 1),
    )
    for q in range(1, int(n_quantiles) + 1):
        is_q = labels == float(q)
        w = weights.where(is_q, 0.0)
        denom = w.sum(axis=1)
        if weighting == "cap":
            num = weighted.where(is_q, 0.0).sum(axis=1)
        else:
            num = fwd.where(is_q).sum(axis=1)
        out[q] = (num / denom.replace(0.0, np.nan)).astype("float64")

    out.columns.name = "quantile"
    return out


def long_short(quantile_returns: pd.DataFrame) -> pd.Series:
    """
    Long-short return: top quantile minus bottom quantile.

    Args:
        quantile_returns: Output of :func:`quantile_returns`.

    Returns:
        Series of ``Q_last - Q_1`` returns indexed by date.

    Raises:
        ValueError: If fewer than two quantile columns are supplied.
    """
    if not isinstance(quantile_returns, pd.DataFrame):
        raise ValueError("quantile_returns must be a pandas DataFrame")
    if quantile_returns.shape[1] < 2:
        raise ValueError("Need at least 2 quantiles")
    return (
        quantile_returns.iloc[:, -1] - quantile_returns.iloc[:, 0]
    ).rename("long_short")


def turnover(
    factor: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """
    Quantile turnover: fraction of assets that change bucket versus the
    previous date.

    ``turnover_t = #{i : q_i,t != q_i,t-1 and both defined} / #{i : q_i,t defined}``

    This is a holdings-churn proxy, not a dollar-volume turnover measure.
    The first date is NaN.

    Args:
        factor: Long factor Series, MultiIndex ``(date, asset)``.
        n_quantiles: Number of buckets (>= 2).

    Returns:
        Series of turnover rates indexed by date, named ``turnover``.
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("factor must have a MultiIndex (date, asset)")
    if not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 2:
        raise ValueError("n_quantiles must be an integer >= 2")

    q = assign_quantiles(factor, n_quantiles).unstack(level="asset")
    prev = q.shift(1)
    changed = (q != prev) & q.notna() & prev.notna()
    valid = q.notna().sum(axis=1).replace(0, np.nan)
    result = (changed.sum(axis=1) / valid).rename("turnover")
    if len(result):
        result.iloc[0] = np.nan  # no previous holdings to compare against
    return result
