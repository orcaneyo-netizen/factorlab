"""
factorlab.report — One-call factor report with IC, quantiles, turnover and DSR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .deflated import DSRResult, deflated_sharpe_ratio
from .evaluate import (
    ICSummary,
    forward_returns,
    ic_summary,
    long_short,
    quantile_returns,
    turnover,
)
from .panel import PanelData


@dataclass
class FactorReport:
    """Container for a factor evaluation with a markdown export."""

    factor_name: str
    horizon: int
    ic_summary: ICSummary
    quantile_returns: pd.DataFrame
    long_short_return: pd.Series
    turnover: pd.Series
    dsr: DSRResult
    quantile_monotonic: bool

    def to_dict(self) -> dict[str, Any]:
        """Flatten the report into JSON-friendly primitives."""
        ls = self.long_short_return.dropna()
        if len(ls) > 1 and ls.std(ddof=1) > 0:
            ls_sharpe = float(ls.mean() / ls.std(ddof=1) * np.sqrt(252.0))
        else:
            ls_sharpe = np.nan
        return {
            "factor_name": self.factor_name,
            "horizon": self.horizon,
            "ic_summary": self.ic_summary.to_dict(),
            "quantile_returns": {
                int(q): float(v) for q, v in self.quantile_returns.mean().items()
            },
            "long_short_mean": float(ls.mean()) if len(ls) else np.nan,
            "long_short_sharpe": ls_sharpe,
            "turnover_mean": float(self.turnover.mean())
            if self.turnover.notna().any()
            else np.nan,
            "dsr": asdict(self.dsr),
            "quantile_monotonic": bool(self.quantile_monotonic),
        }

    def to_markdown(self) -> str:
        """Render the report as a compact markdown document."""
        d = self.to_dict()
        ic = d["ic_summary"]
        qr = d["quantile_returns"]

        def fmt(value: float, digits: int = 4) -> str:
            return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

        lines = [
            f"# Factor Report: {self.factor_name}",
            "",
            f"**Horizon:** {self.horizon} day(s)  ",
            f"**Quantile monotonic (Q1 -> Q{len(qr)}):** "
            f"{'yes' if self.quantile_monotonic else 'no'}",
            "",
            "## IC Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean IC | {fmt(ic['mean_ic'])} |",
            f"| IC IR | {fmt(ic['ic_ir'])} |",
            f"| t-stat (Newey-West) | {fmt(ic['t_stat'])} |",
            f"| Hit rate | {fmt(ic['hit_rate'])} |",
            f"| Observations | {ic['n_obs']} |",
            "",
            "### IC decay",
            "",
            "| Horizon | Mean IC |",
            "|---------|---------|",
        ]
        for h, value in ic["ic_decay"].items():
            lines.append(f"| {h} | {fmt(value)} |")

        lines.extend(
            [
                "",
                "## Quantile returns (equal weight)",
                "",
                "| Quantile | Mean return |",
                "|----------|-------------|",
            ]
        )
        for q in sorted(qr):
            lines.append(f"| Q{q} | {fmt(qr[q], 6)} |")

        lines.extend(
            [
                "",
                f"**Long-short (Q{len(qr)} - Q1):** {fmt(d['long_short_mean'], 6)}  ",
                f"**Long-short Sharpe (annualized):** {fmt(d['long_short_sharpe'])}",
                "",
                "## Turnover",
                "",
                f"Mean quantile turnover: **{fmt(d['turnover_mean'])}**",
                "",
                "## Deflated Sharpe Ratio",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| DSR | {fmt(d['dsr']['dsr'])} |",
                f"| PSR | {fmt(d['dsr']['psr'])} |",
                f"| Null E[max SR] (per period) | {fmt(d['dsr']['sr_null'], 6)} |",
                f"| Trials | {d['dsr']['n_trials']} |",
                f"| Observations | {d['dsr']['n_obs']} |",
            ]
        )
        return "\n".join(lines)


def factor_report(
    panel: PanelData,
    factor: pd.Series,
    fwd_horizon: int = 1,
    n_quantiles: int = 5,
    n_trials: int = 100,
) -> FactorReport:
    """
    Build a complete factor report.

    The DSR is computed on the daily long-short portfolio returns using a
    **per-period** Sharpe ratio (``mean / std``), which is the scale expected
    by :func:`factorlab.deflated.deflated_sharpe_ratio`.

    Args:
        panel: Validated :class:`PanelData`.
        factor: Long factor Series, MultiIndex ``(date, asset)``.
        fwd_horizon: Forward return horizon in days (>= 1).
        n_quantiles: Number of quantile portfolios (>= 2).
        n_trials: Number of trials used by the DSR (>= 1).

    Returns:
        :class:`FactorReport` with a ``to_markdown()`` method.
    """
    if not isinstance(panel, PanelData):
        raise ValueError("panel must be a PanelData instance")
    if not isinstance(factor, pd.Series):
        raise ValueError("factor must be a pandas Series")
    if not isinstance(fwd_horizon, (int, np.integer)) or fwd_horizon < 1:
        raise ValueError("fwd_horizon must be an integer >= 1")
    if not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 2:
        raise ValueError("n_quantiles must be an integer >= 2")
    if not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1")

    fwd = forward_returns(panel.returns, int(fwd_horizon))
    ic_summ = ic_summary(factor, panel.returns, max_horizon=10)

    q_returns = quantile_returns(
        factor, fwd, n_quantiles=int(n_quantiles), weighting="equal"
    )
    ls_return = long_short(q_returns)
    churn = turnover(factor, int(n_quantiles))

    ls_clean = ls_return.dropna()
    if len(ls_clean) > 1:
        ls_std = float(ls_clean.std(ddof=1))
        sr_period = float(ls_clean.mean() / ls_std) if ls_std > 0 else 0.0
        skew = float(stats.skew(ls_clean, bias=False)) if len(ls_clean) > 2 else 0.0
        kurt = (
            float(stats.kurtosis(ls_clean, fisher=False, bias=False))
            if len(ls_clean) > 3
            else 3.0
        )
        dsr = deflated_sharpe_ratio(
            sr_period, int(n_trials), len(ls_clean), skew, kurt
        )
    else:
        dsr = deflated_sharpe_ratio(0.0, int(n_trials), 2, 0.0, 3.0)

    q_means = q_returns.mean().dropna()
    monotonic = bool(
        len(q_means) >= 2
        and all(
            q_means.iloc[i] <= q_means.iloc[i + 1]
            for i in range(len(q_means) - 1)
        )
    )

    return FactorReport(
        factor_name=str(factor.name) if factor.name is not None else "factor",
        horizon=int(fwd_horizon),
        ic_summary=ic_summ,
        quantile_returns=q_returns,
        long_short_return=ls_return,
        turnover=churn,
        dsr=dsr,
        quantile_monotonic=monotonic,
    )
