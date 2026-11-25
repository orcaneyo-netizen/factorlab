"""
factorlab — a small, offline cross-sectional equity factor research lab.

Quickstart::

    from factorlab import synthetic_panel, PanelData, factors, evaluate

    panel = PanelData(synthetic_panel(n_assets=40, n_days=750, seed=42))
    mom = factors.momentum(panel.close)
    fwd = evaluate.forward_returns(panel.returns, horizon=5)
    print(evaluate.ic_summary(mom, panel.returns).to_dict())
"""

from __future__ import annotations

from . import cv, deflated, evaluate, factors, panel, preprocess, report
from .cv import PurgedKFold, purged_cv_splits
from .deflated import (
    DSRResult,
    PBOResult,
    cscv_combinations,
    cscv_paths_from_returns,
    cscv_paths_from_splits,
    deflated_sharpe_ratio,
    pbo,
)
from .evaluate import (
    ICSummary,
    assign_quantiles,
    forward_returns,
    ic_series,
    ic_summary,
    long_short,
    quantile_returns,
    turnover,
)
from .factors import (
    amihud,
    liquidity,
    momentum,
    short_term_reversal,
    trend_quality,
    volatility,
)
from .panel import PanelData, from_wide, synthetic_panel, to_long, to_wide, validate_panel
from .preprocess import (
    demean,
    fillna_cross_section,
    neutralize,
    rank_normal,
    winsorize,
    zscore,
)
from .report import FactorReport, factor_report

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # modules
    "panel",
    "preprocess",
    "factors",
    "evaluate",
    "cv",
    "deflated",
    "report",
    # panel
    "PanelData",
    "synthetic_panel",
    "validate_panel",
    "to_wide",
    "to_long",
    "from_wide",
    # preprocess
    "winsorize",
    "zscore",
    "rank_normal",
    "demean",
    "neutralize",
    "fillna_cross_section",
    # factors
    "momentum",
    "short_term_reversal",
    "volatility",
    "liquidity",
    "amihud",
    "trend_quality",
    # evaluate
    "forward_returns",
    "ic_series",
    "ic_summary",
    "ICSummary",
    "assign_quantiles",
    "quantile_returns",
    "long_short",
    "turnover",
    # cv
    "PurgedKFold",
    "purged_cv_splits",
    # deflated
    "deflated_sharpe_ratio",
    "DSRResult",
    "pbo",
    "PBOResult",
    "cscv_combinations",
    "cscv_paths_from_returns",
    "cscv_paths_from_splits",
    # report
    "factor_report",
    "FactorReport",
]
