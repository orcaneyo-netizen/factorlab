"""DSR and PBO tests."""

from __future__ import annotations

import numpy as np
import pytest

from factorlab.deflated import (
    cscv_combinations,
    cscv_paths_from_returns,
    cscv_paths_from_splits,
    deflated_sharpe_ratio,
    pbo,
)


def test_dsr_decreases_with_more_trials() -> None:
    values = [
        deflated_sharpe_ratio(sr=0.05, n_trials=n, n_obs=500).dsr
        for n in (1, 10, 100, 1000)
    ]
    assert values[0] > values[1] > values[2] > values[3]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_dsr_null_threshold_increases_with_trials() -> None:
    nulls = [
        deflated_sharpe_ratio(sr=0.05, n_trials=n, n_obs=500).sr_null
        for n in (2, 10, 100, 1000)
    ]
    assert nulls == sorted(nulls)
    assert nulls[0] > 0.0
    assert deflated_sharpe_ratio(sr=0.05, n_trials=1, n_obs=500).sr_null == 0.0


def test_dsr_increases_with_observed_sharpe() -> None:
    low = deflated_sharpe_ratio(sr=0.01, n_trials=50, n_obs=300)
    high = deflated_sharpe_ratio(sr=0.20, n_trials=50, n_obs=300)
    assert high.dsr > low.dsr
    assert high.psr > low.psr
    assert 0.0 <= low.dsr <= 1.0
    assert 0.0 <= high.dsr <= 1.0


def test_dsr_result_fields_and_moments() -> None:
    result = deflated_sharpe_ratio(
        sr=0.1, n_trials=20, n_obs=250, skew=-0.5, kurt=5.0
    )
    assert result.n_trials == 20
    assert result.n_obs == 250
    assert result.skew == -0.5
    assert result.kurt == 5.0
    assert np.isfinite(result.dsr)
    assert np.isfinite(result.psr)
    assert np.isfinite(result.sr_null)


def test_dsr_validation() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        deflated_sharpe_ratio(sr=0.1, n_trials=5, n_obs=1)
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(sr=0.1, n_trials=0, n_obs=100)
    with pytest.raises(ValueError, match="finite"):
        deflated_sharpe_ratio(sr=np.nan, n_trials=5, n_obs=100)
    with pytest.raises(ValueError, match="moment"):
        deflated_sharpe_ratio(sr=1.0, n_trials=5, n_obs=100, skew=20.0, kurt=3.0)


def test_cscv_combinations_count_and_symmetry() -> None:
    combos = cscv_combinations(10)
    assert len(combos) == 252  # C(10, 5)
    for is_blocks, oos_blocks in combos:
        assert len(is_blocks) == len(oos_blocks) == 5
        assert set(is_blocks).isdisjoint(oos_blocks)
        assert set(is_blocks) | set(oos_blocks) == set(range(10))


def test_cscv_combinations_validation() -> None:
    with pytest.raises(ValueError, match="even"):
        cscv_combinations(9)
    with pytest.raises(ValueError, match=">= 2"):
        cscv_combinations(0)


def test_pbo_in_unit_interval() -> None:
    rng = np.random.default_rng(7)
    paths = cscv_paths_from_returns(rng.normal(size=400), n_blocks=10)
    result = pbo(paths)
    assert 0.0 <= result.pbo <= 1.0
    assert np.isfinite(result.logit)
    assert result.n_paths == 252
    assert result.n_strategies == 1


def test_pbo_lower_for_predictive_factor_than_noise() -> None:
    rng = np.random.default_rng(8)
    n_obs = 600
    noise = rng.normal(0.0, 0.01, n_obs)
    predictive = 0.002 + rng.normal(0.0, 0.01, n_obs)

    pbo_noise = pbo(cscv_paths_from_returns(noise, n_blocks=10)).pbo
    pbo_predictive = pbo(cscv_paths_from_returns(predictive, n_blocks=10)).pbo

    assert pbo_predictive < pbo_noise
    assert pbo_predictive < 0.05
    assert 0.25 < pbo_noise < 0.75


def test_pbo_single_strategy_rule() -> None:
    good = [{"is_sr": 1.0, "oos_sr": 0.5}] * 10
    bad = [{"is_sr": 1.0, "oos_sr": -0.5}] * 10
    assert pbo(good).pbo == 0.0
    assert pbo(bad).pbo == 1.0


def test_pbo_multi_strategy_rank_rule() -> None:
    # In-sample best is the third strategy; out of sample it is the worst.
    overfit = [{"is_sr": [1.0, 2.0, 3.0], "oos_sr": [3.0, 2.0, 1.0]}] * 4
    assert pbo(overfit).pbo == 1.0

    # In-sample best stays the best out of sample.
    honest = [{"is_sr": [3.0, 2.0, 1.0], "oos_sr": [3.0, 2.0, 1.0]}] * 4
    assert pbo(honest).pbo == 0.0
    assert pbo(honest).n_strategies == 3


def test_pbo_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        pbo([])
    with pytest.raises(ValueError, match="mapping"):
        pbo([1, 2])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="keys"):
        pbo([{"foo": 1.0}])
    with pytest.raises(ValueError, match="same number"):
        pbo([{"is_sr": [1.0, 2.0], "oos_sr": [1.0]}, {"is_sr": [1.0], "oos_sr": [1.0]}])
    with pytest.raises(ValueError, match="finite"):
        pbo([{"is_sr": np.nan, "oos_sr": 0.0}])


def test_cscv_paths_from_splits_uses_purged_folds() -> None:
    rng = np.random.default_rng(9)
    returns = 0.001 + rng.normal(0.0, 0.01, 300)
    splits = [
        (np.arange(0, 100), np.arange(120, 200)),
        (np.arange(200, 300), np.arange(100, 120)),
    ]
    paths = cscv_paths_from_splits(returns, splits)
    assert len(paths) == 2
    result = pbo(paths)
    assert 0.0 <= result.pbo <= 1.0
    assert result.n_paths == 2


def test_cscv_returns_validation() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        cscv_paths_from_returns(np.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match="at least two"):
        cscv_paths_from_returns(np.array([0.01]))
    with pytest.raises(ValueError, match="n_obs"):
        cscv_paths_from_returns(np.zeros(5), n_blocks=10)
