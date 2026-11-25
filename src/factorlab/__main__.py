"""
factorlab.__main__ — command line entry point.

Usage::

    python -m factorlab demo
    python -m factorlab demo --assets 40 --days 750 --seed 42 --horizon 5
    python -m factorlab demo --plot factorlab_demo.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .deflated import deflated_sharpe_ratio
from .evaluate import (
    forward_returns,
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
from .panel import PanelData, synthetic_panel


def _build_factors(panel: PanelData) -> dict[str, pd.Series]:
    """Compute the full factor library on a panel."""
    return {
        "momentum": momentum(panel.close, lookback=126, skip=21),
        "reversal": short_term_reversal(panel.returns, window=5),
        "low_vol": volatility(panel.returns, window=60),
        "liquidity": liquidity(panel.volume, panel.market_cap),
        "amihud": amihud(panel.returns, panel.volume, window=21),
        "trend_quality": trend_quality(panel.close, window=126),
    }


def _fmt(value: float, digits: int = 3) -> str:
    """Format a float for the console, tolerating NaN."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}f}"


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def run_demo(
    n_assets: int = 40,
    n_days: int = 750,
    seed: int = 42,
    horizon: int = 5,
    n_quantiles: int = 5,
    plot_path: str | None = None,
) -> int:
    """
    Run the end-to-end synthetic example and print a compact report.

    Args:
        n_assets: Number of synthetic assets.
        n_days: Number of synthetic trading days.
        seed: Random seed.
        horizon: Forward-return horizon for the detailed section.
        n_quantiles: Number of quantile portfolios.
        plot_path: If given, save a diagnostic PNG to this path.

    Returns:
        Process exit code (0 on success).
    """
    panel = PanelData(synthetic_panel(n_assets=n_assets, n_days=n_days, seed=seed))
    factors = _build_factors(panel)

    print("=" * 78)
    print("factorlab demo — synthetic cross-sectional equity panel")
    print("=" * 78)
    print(
        f"panel: {len(panel.assets)} assets x {len(panel.dates)} days "
        f"({panel.dates[0].date()} .. {panel.dates[-1].date()}), seed={seed}"
    )

    _print_header("Factor comparison (IC at horizon 1)")
    header = (
        f"{'factor':<14}{'mean IC':>10}{'IC IR':>9}{'NW t':>9}"
        f"{'hit':>8}{'IC@5':>9}{'n':>6}"
    )
    print(header)
    print("-" * len(header))
    summaries = {}
    for name, factor in factors.items():
        summary = ic_summary(factor, panel.returns, max_horizon=10, nw_lags=5)
        summaries[name] = summary
        ic5 = summary.ic_decay.get(5, np.nan)
        print(
            f"{name:<14}{_fmt(summary.mean_ic):>10}{_fmt(summary.ic_ir):>9}"
            f"{_fmt(summary.t_stat):>9}{_fmt(summary.hit_rate, 2):>8}"
            f"{_fmt(ic5):>9}{summary.n_obs:>6}"
        )

    # Detailed section for momentum.
    mom = factors["momentum"]
    fwd = forward_returns(panel.returns, horizon)
    q_ret = quantile_returns(mom, fwd, n_quantiles=n_quantiles, weighting="equal")
    ls = long_short(q_ret)
    churn = turnover(mom, n_quantiles)

    _print_header(f"Momentum detail (horizon={horizon}, {n_quantiles} quantiles)")
    print(f"{'quantile':<10}{'mean fwd return':>18}{'ann. vol':>12}")
    print("-" * 40)
    for q in q_ret.columns:
        series = q_ret[q].dropna()
        vol = series.std(ddof=1) * np.sqrt(252.0 / max(horizon, 1)) if len(series) > 1 else np.nan
        print(f"Q{int(q):<9}{_fmt(series.mean(), 5):>18}{_fmt(vol, 4):>12}")

    ls_clean = ls.dropna()
    if len(ls_clean) > 1 and ls_clean.std(ddof=1) > 0:
        sr_period = float(ls_clean.mean() / ls_clean.std(ddof=1))
        ls_sharpe = sr_period * np.sqrt(252.0 / max(horizon, 1))
        dsr = deflated_sharpe_ratio(
            sr_period,
            n_trials=len(factors) * 10,
            n_obs=len(ls_clean),
            skew=float(pd.Series(ls_clean).skew()),
            kurt=float(pd.Series(ls_clean).kurtosis()) + 3.0,
        )
        dsr_text = f"{dsr.dsr:.3f} (PSR {dsr.psr:.3f}, null SR {dsr.sr_null:.4f})"
    else:
        ls_sharpe = np.nan
        dsr_text = "n/a"

    print()
    print(f"long-short mean      : {_fmt(ls_clean.mean(), 5)}")
    print(f"long-short Sharpe ann: {_fmt(ls_sharpe)}")
    print(f"mean quantile turnover: {_fmt(churn.mean(), 3)}")
    print(f"deflated Sharpe      : {dsr_text}")

    _print_header("IC decay (mean IC, horizons 1..10)")
    horizons = list(range(1, 11))
    print("factor".ljust(14) + "".join(f"{h:>7}" for h in horizons))
    print("-" * (14 + 7 * len(horizons)))
    for name, summary in summaries.items():
        row = "".join(f"{_fmt(summary.ic_decay.get(h, np.nan), 2):>7}" for h in horizons)
        print(f"{name:<14}{row}")

    if plot_path:
        try:
            _save_plot(plot_path, summaries, q_ret, ls)
        except ImportError:  # pragma: no cover - matplotlib is a declared dep
            print("\nmatplotlib is not available; skipping --plot")
            return 1
        print(f"\nsaved diagnostic plot to {plot_path}")

    print()
    print("done.")
    return 0


def _save_plot(
    path: str,
    summaries: dict,
    q_ret: pd.DataFrame,
    ls: pd.Series,
) -> None:
    """Save a two-panel diagnostic figure (headless Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, summary in summaries.items():
        axes[0].plot(
            summary.ic_decay.index,
            summary.ic_decay.to_numpy(dtype="float64"),
            marker="o",
            label=name,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("IC decay")
    axes[0].set_xlabel("horizon (days)")
    axes[0].set_ylabel("mean IC")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    cumulative = (1.0 + q_ret.fillna(0.0)).cumprod()
    for q in cumulative.columns:
        axes[1].plot(cumulative.index, cumulative[q], label=f"Q{int(q)}")
    axes[1].set_title("Momentum quantile portfolios (cumulative)")
    axes[1].set_ylabel("growth of 1")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="factorlab",
        description="Cross-sectional equity factor research lab (offline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run an end-to-end synthetic example")
    demo.add_argument("--assets", type=int, default=40, help="number of assets")
    demo.add_argument("--days", type=int, default=750, help="number of trading days")
    demo.add_argument("--seed", type=int, default=42, help="random seed")
    demo.add_argument("--horizon", type=int, default=5, help="forward horizon (days)")
    demo.add_argument("--quantiles", type=int, default=5, help="number of quantiles")
    demo.add_argument(
        "--plot",
        nargs="?",
        const="factorlab_demo.png",
        default=None,
        help="save a diagnostic PNG instead of only printing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        try:
            return run_demo(
                n_assets=args.assets,
                n_days=args.days,
                seed=args.seed,
                horizon=args.horizon,
                n_quantiles=args.quantiles,
                plot_path=args.plot,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    parser.error(f"unknown command {args.command!r}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
