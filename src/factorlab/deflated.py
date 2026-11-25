"""
factorlab.deflated — Multiple-testing guards for backtest Sharpe ratios.

Two tools from Bailey, Borwein, López de Prado & Zhu:

* **Deflated Sharpe Ratio (DSR)** — the probability that the observed Sharpe
  ratio beats the expected maximum Sharpe ratio of ``n_trials`` independent
  zero-skill strategies, accounting for sample length, skewness and kurtosis.
* **Probability of Backtest Overfitting (PBO)** — the Combinatorially
  Symmetric Cross-Validation (CSCV) estimate of how often the in-sample best
  strategy underperforms out of sample.

Sharpe ratios in this module are **per-observation** (not annualized).  Divide
an annualized Sharpe by ``sqrt(252)`` before calling :func:`deflated_sharpe_ratio`
when working with daily returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DSRResult:
    """Output of :func:`deflated_sharpe_ratio`."""

    dsr: float  # P(SR > E[max SR | no skill]) under the null
    psr: float  # P(SR > 0) — probabilistic Sharpe ratio
    sr_null: float  # E[max Sharpe] over n_trials zero-skill strategies
    n_trials: int
    n_obs: int
    skew: float
    kurt: float


def deflated_sharpe_ratio(
    sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> DSRResult:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Under the null of zero skill, the maximum Sharpe across ``N`` independent
    trials has expectation::

        SR_0 = sqrt(Var(SR)) * [ (1 - g) * Z^{-1}(1 - 1/N)
                                 + g * Z^{-1}(1 - 1/(N e)) ]

    with ``g`` the Euler-Mascheroni constant and ``Var(SR) = 1 / (n_obs - 1)``.
    The probabilistic Sharpe ratio with a non-zero benchmark is::

        PSR(SR*) = Phi( (SR - SR*) * sqrt(n_obs - 1)
                        / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )

    ``PSR`` uses ``SR* = 0``; ``DSR`` uses ``SR* = SR_0``.  A DSR below ~0.95
    means the result is not convincing after accounting for the number of
    trials.

    Args:
        sr: Observed Sharpe ratio, per observation (not annualized).
        n_trials: Number of independent strategies/configurations tried (>= 1).
        n_obs: Number of return observations used for ``sr`` (>= 2).
        skew: Sample skewness of returns (0 for normal).
        kurt: Sample kurtosis of returns, non-excess (3 for normal).

    Returns:
        :class:`DSRResult` with ``dsr``, ``psr`` and ``sr_null``.

    Raises:
        ValueError: On invalid sample sizes or moment combinations.
    """
    if not isinstance(n_obs, (int, np.integer)) or n_obs < 2:
        raise ValueError("n_obs must be an integer >= 2")
    if not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1")
    if not np.isfinite(sr):
        raise ValueError("sr must be finite")
    if not np.isfinite(skew) or not np.isfinite(kurt):
        raise ValueError("skew and kurt must be finite")

    n_obs = int(n_obs)
    n_trials = int(n_trials)

    # Sampling variance of the Sharpe ratio under the null.
    var_sr = 1.0 / (n_obs - 1)

    if n_trials == 1:
        sr_null = 0.0
    else:
        sr_null = np.sqrt(var_sr) * (
            (1.0 - _EULER_MASCHERONI)
            * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + _EULER_MASCHERONI
            * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )

    # Variance of the Sharpe estimator with non-normal returns (Mertens).
    denom_var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if not np.isfinite(denom_var) or denom_var <= 0.0:
        raise ValueError(
            "invalid moment combination: 1 - skew*sr + (kurt-1)/4*sr^2 <= 0"
        )
    se_sr = np.sqrt(denom_var / (n_obs - 1))

    psr = float(stats.norm.cdf(sr / se_sr))
    dsr = float(stats.norm.cdf((sr - sr_null) / se_sr))

    return DSRResult(
        dsr=dsr,
        psr=psr,
        sr_null=float(sr_null),
        n_trials=n_trials,
        n_obs=n_obs,
        skew=float(skew),
        kurt=float(kurt),
    )


@dataclass(frozen=True)
class PBOResult:
    """Output of :func:`pbo`."""

    pbo: float  # Probability of backtest overfitting, in [0, 1]
    logit: float  # logit(pbo), clipped away from +/-inf
    n_paths: int  # Number of CSCV paths
    n_strategies: int  # Number of strategies compared per path
    mean_is_sr: float  # Mean in-sample Sharpe across paths
    mean_oos_sr: float  # Mean out-of-sample Sharpe across paths


def _as_matrix(values: Sequence[object]) -> np.ndarray:
    """Coerce a sequence of scalars/arrays to a 2-D float array."""
    rows = [np.atleast_1d(np.asarray(v, dtype="float64")) for v in values]
    width = rows[0].size
    if any(row.size != width for row in rows):
        raise ValueError("every CSCV path must have the same number of strategies")
    return np.vstack(rows)


def _extract_paths(
    cscv_paths: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize CSCV path dicts to ``(is_sr, oos_sr)`` matrices."""
    if isinstance(cscv_paths, (str, bytes)) or not isinstance(cscv_paths, Sequence):
        raise ValueError("cscv_paths must be a non-empty sequence of mappings")
    if len(cscv_paths) == 0:
        raise ValueError("cscv_paths cannot be empty")

    is_values: list[object] = []
    oos_values: list[object] = []
    for i, path in enumerate(cscv_paths):
        if not isinstance(path, Mapping):
            raise ValueError(f"cscv_paths[{i}] must be a mapping")
        if "is_sr" in path and "oos_sr" in path:
            is_values.append(path["is_sr"])
            oos_values.append(path["oos_sr"])
        elif "train_sr" in path and "test_sr" in path:
            is_values.append(path["train_sr"])
            oos_values.append(path["test_sr"])
        else:
            raise ValueError(
                "each path needs 'is_sr'/'oos_sr' (or legacy "
                "'train_sr'/'test_sr') keys"
            )

    is_mat = _as_matrix(is_values)
    oos_mat = _as_matrix(oos_values)
    if not np.isfinite(is_mat).all() or not np.isfinite(oos_mat).all():
        raise ValueError("CSCV Sharpe ratios must be finite")
    return is_mat, oos_mat


def pbo(cscv_paths: Sequence[Mapping[str, object]]) -> PBOResult:
    """
    Probability of Backtest Overfitting via CSCV.

    For every CSCV path the in-sample best strategy is selected; overfitting
    occurs when that strategy lands in the bottom half of the out-of-sample
    ranking.  With a single strategy the rule reduces to "out-of-sample
    Sharpe <= 0", i.e. the probability that the strategy fails after costs of
    selection.

    Args:
        cscv_paths: Sequence of mappings, each holding either
            ``{'is_sr': ..., 'oos_sr': ...}`` or the legacy
            ``{'train_sr': ..., 'test_sr': ...}``.  Values may be scalars
            (one strategy) or 1-D arrays (several strategies per path).

    Returns:
        :class:`PBOResult`.

    Raises:
        ValueError: If paths are empty, malformed, ragged, or non-finite.
    """
    is_mat, oos_mat = _extract_paths(cscv_paths)
    n_paths, n_strategies = is_mat.shape

    if n_strategies == 1:
        overfit = oos_mat[:, 0] <= 0.0
    else:
        best = np.argmax(is_mat, axis=1)
        # Dense rank of each strategy's OOS Sharpe within its path (1 = worst).
        order = np.argsort(np.argsort(oos_mat, axis=1), axis=1)
        best_rank = order[np.arange(n_paths), best] + 1
        overfit = best_rank <= n_strategies / 2.0

    pbo_value = float(np.mean(overfit))
    eps = 1e-12
    clipped = float(np.clip(pbo_value, eps, 1.0 - eps))
    logit = float(np.log(clipped / (1.0 - clipped)))

    return PBOResult(
        pbo=pbo_value,
        logit=logit,
        n_paths=n_paths,
        n_strategies=n_strategies,
        mean_is_sr=float(is_mat.mean()),
        mean_oos_sr=float(oos_mat.mean()),
    )


def cscv_combinations(n_blocks: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Enumerate all symmetric IS/OOS block partitions for CSCV.

    Args:
        n_blocks: Even number of contiguous time blocks (>= 2).

    Returns:
        List of ``(in_sample_blocks, out_of_sample_blocks)`` index tuples.

    Raises:
        ValueError: If ``n_blocks`` is odd or smaller than 2.
    """
    if not isinstance(n_blocks, (int, np.integer)) or n_blocks < 2:
        raise ValueError("n_blocks must be an integer >= 2")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even for symmetric CSCV")

    all_blocks = tuple(range(int(n_blocks)))
    half = int(n_blocks) // 2
    paths: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for combo in combinations(all_blocks, half):
        complement = tuple(b for b in all_blocks if b not in combo)
        paths.append((combo, complement))
    return paths


def _sharpe(returns: np.ndarray, annualize: float) -> np.ndarray:
    """Column-wise Sharpe ratio; NaN for columns with < 2 obs or zero variance."""
    returns = np.asarray(returns, dtype="float64")
    if returns.ndim == 1:
        returns = returns[:, None]
    n = returns.shape[0]
    if n < 2:
        return np.full(returns.shape[1], np.nan)
    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = mean / std * np.sqrt(annualize)
    sr[~np.isfinite(sr)] = np.nan
    return sr


def cscv_paths_from_returns(
    returns: np.ndarray,
    n_blocks: int = 10,
    annualize: float = 252.0,
) -> list[dict[str, np.ndarray]]:
    """
    Build CSCV paths from a matrix of strategy returns.

    The sample is cut into ``n_blocks`` contiguous blocks; every way of
    choosing half of them forms an in-sample set, with the complement used
    out of sample.  Each path reports the per-strategy Sharpe ratio on both
    sides.

    Args:
        returns: Array ``(n_obs,)`` or ``(n_obs, n_strategies)`` of returns.
        n_blocks: Even number of blocks (>= 2).
        annualize: Annualization factor (252 for daily data).

    Returns:
        List of ``{'is_sr': ndarray, 'oos_sr': ndarray}`` dicts, one per
        combination (``C(n_blocks, n_blocks/2)`` paths).

    Raises:
        ValueError: On a bad shape, too few observations, or invalid blocks.
    """
    returns = np.asarray(returns, dtype="float64")
    if returns.ndim == 1:
        returns = returns[:, None]
    if returns.ndim != 2:
        raise ValueError("returns must be 1-D or 2-D")
    n_obs = returns.shape[0]
    if n_obs < 2:
        raise ValueError("returns must contain at least two observations")
    if not np.isfinite(returns).all():
        raise ValueError("returns must be finite")
    combos = cscv_combinations(n_blocks)
    if n_obs < n_blocks:
        raise ValueError("n_obs must be >= n_blocks")

    blocks = np.array_split(np.arange(n_obs), int(n_blocks))
    paths: list[dict[str, np.ndarray]] = []
    for is_blocks, oos_blocks in combos:
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])
        paths.append(
            {
                "is_sr": _sharpe(returns[is_idx], annualize),
                "oos_sr": _sharpe(returns[oos_idx], annualize),
            }
        )
    return paths


def cscv_paths_from_splits(
    returns: np.ndarray,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    annualize: float = 252.0,
) -> list[dict[str, float]]:
    """
    Build single-strategy CSCV paths from arbitrary ``(train, test)`` splits.

    Useful with :class:`factorlab.cv.PurgedKFold`: each split contributes one
    in-sample / out-of-sample Sharpe pair.

    Args:
        returns: 1-D array of strategy returns.
        cv_splits: Sequence of ``(train_idx, test_idx)`` positional arrays.
        annualize: Annualization factor.

    Returns:
        List of ``{'is_sr': float, 'oos_sr': float}`` dicts.
    """
    returns = np.asarray(returns, dtype="float64").ravel()
    if returns.size < 2:
        raise ValueError("returns must contain at least two observations")
    paths: list[dict[str, float]] = []
    for train_idx, test_idx in cv_splits:
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        paths.append(
            {
                "is_sr": float(_sharpe(returns[train_idx], annualize)[0]),
                "oos_sr": float(_sharpe(returns[test_idx], annualize)[0]),
            }
        )
    if not paths:
        raise ValueError("no non-empty splits were supplied")
    return paths
