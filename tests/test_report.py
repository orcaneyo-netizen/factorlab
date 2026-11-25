"""Report and CLI tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlab.__main__ import main, run_demo
from factorlab.factors import momentum
from factorlab.panel import PanelData
from factorlab.report import factor_report


def test_factor_report_structure(panel: PanelData) -> None:
    factor = momentum(panel.close, lookback=40, skip=5)
    report = factor_report(
        panel, factor, fwd_horizon=5, n_quantiles=5, n_trials=20
    )

    assert report.factor_name == "momentum"
    assert report.horizon == 5
    assert report.quantile_returns.shape[1] == 5
    assert isinstance(report.quantile_monotonic, bool)
    assert 0.0 <= report.dsr.dsr <= 1.0
    assert report.dsr.n_trials == 20
    assert report.turnover.index.equals(panel.dates)

    data = report.to_dict()
    for key in (
        "factor_name",
        "horizon",
        "ic_summary",
        "quantile_returns",
        "long_short_mean",
        "long_short_sharpe",
        "turnover_mean",
        "dsr",
        "quantile_monotonic",
    ):
        assert key in data
    assert set(data["quantile_returns"]) == {1, 2, 3, 4, 5}
    assert len(data["ic_summary"]["ic_decay"]) == 10


def test_factor_report_markdown(panel: PanelData) -> None:
    factor = momentum(panel.close, lookback=40, skip=5)
    md = factor_report(panel, factor, fwd_horizon=1).to_markdown()
    assert "# Factor Report: momentum" in md
    assert "## IC Summary" in md
    assert "### IC decay" in md
    assert "## Quantile returns" in md
    assert "## Deflated Sharpe Ratio" in md
    assert "| Q1 |" in md
    assert "nan" not in md.lower()


def test_factor_report_validation(panel: PanelData) -> None:
    factor = momentum(panel.close, lookback=40, skip=5)
    with pytest.raises(ValueError, match="PanelData"):
        factor_report(panel.long, factor)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fwd_horizon"):
        factor_report(panel, factor, fwd_horizon=0)
    with pytest.raises(ValueError, match="n_quantiles"):
        factor_report(panel, factor, n_quantiles=1)
    with pytest.raises(ValueError, match="n_trials"):
        factor_report(panel, factor, n_trials=0)


def test_cli_demo_runs(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "demo",
            "--assets",
            "10",
            "--days",
            "220",
            "--seed",
            "3",
            "--horizon",
            "2",
            "--quantiles",
            "4",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "factorlab demo" in out
    assert "Factor comparison" in out
    assert "Momentum detail" in out
    assert "IC decay" in out
    assert "done." in out


def test_cli_error_path(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["demo", "--assets", "1", "--days", "30"])
    err = capsys.readouterr().err
    assert code == 2
    assert "error:" in err


def test_run_demo_with_plot(tmp_path) -> None:
    plot = tmp_path / "factorlab_demo.png"
    code = run_demo(n_assets=10, n_days=220, seed=5, horizon=2, plot_path=str(plot))
    assert code == 0
    assert plot.exists()
    assert plot.stat().st_size > 1000
