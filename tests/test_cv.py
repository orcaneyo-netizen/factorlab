"""Purged K-Fold tests: leakage, coverage and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlab.cv import PurgedKFold, purged_cv_splits


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


@pytest.mark.parametrize("n_splits,purge,embargo,horizon", [(5, 2, 1, 3), (4, 0, 0, 1), (3, 5, 2, 10)])
def test_no_label_overlap_and_clearance(
    n_splits: int, purge: int, embargo: int, horizon: int
) -> None:
    n = 60
    dates = _dates(n)
    cv = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo)
    splits = list(cv.split(dates, horizon=horizon))
    assert len(splits) == n_splits

    all_test: list[int] = []
    for train_idx, test_idx in splits:
        assert test_idx.size > 0
        assert train_idx.size > 0
        assert np.intersect1d(train_idx, test_idx).size == 0
        assert np.all(np.diff(train_idx) > 0)

        start, end = int(test_idx[0]), int(test_idx[-1])
        assert np.array_equal(test_idx, np.arange(start, end + 1))

        for i in train_idx:
            # No training label [i, i+horizon-1] may overlap the test window.
            label_end = int(i) + horizon - 1
            label_start = int(i)
            assert label_end < start or label_start > end

            # Purge + embargo clearance in positional distance.
            distance = start - int(i) if i < start else int(i) - end
            assert distance > purge + embargo

        all_test.extend(test_idx.tolist())

    # Test folds partition the full usable range exactly once.
    assert sorted(all_test) == list(range(n))
    assert len(all_test) == n


def test_fold_sizes_sum_to_usable_range() -> None:
    n = 53
    n_splits = 5
    cv = PurgedKFold(n_splits=n_splits, purge=1, embargo=1)
    splits = list(cv.split(_dates(n), horizon=2))
    sizes = [len(test) for _, test in splits]
    fold = n // n_splits
    assert sum(sizes) == n
    assert sizes[:-1] == [fold] * (n_splits - 1)
    assert sizes[-1] == n - fold * (n_splits - 1)


def test_last_fold_absorbs_remainder() -> None:
    n = 47
    cv = PurgedKFold(n_splits=4, purge=0, embargo=0)
    splits = list(cv.split(_dates(n), horizon=1))
    test_indices = np.concatenate([test for _, test in splits])
    assert np.array_equal(test_indices, np.arange(n))
    assert len(splits[-1][1]) == n - 3 * (n // 4)


def test_purged_cv_splits_helper() -> None:
    splits = purged_cv_splits(n_samples=30, n_splits=3, purge=2, embargo=1, horizon=2)
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert train_idx.dtype == np.int64 or train_idx.dtype == int
        assert test_idx.dtype == np.int64 or test_idx.dtype == int


def test_validation_errors() -> None:
    dates = _dates(10)
    with pytest.raises(ValueError, match="n_splits"):
        PurgedKFold(n_splits=1, purge=0)
    with pytest.raises(ValueError, match="purge"):
        PurgedKFold(n_splits=2, purge=-1)
    with pytest.raises(ValueError, match="embargo"):
        PurgedKFold(n_splits=2, purge=0, embargo=-1)
    with pytest.raises(ValueError, match="integer"):
        PurgedKFold(n_splits=2.5, purge=0)  # type: ignore[arg-type]

    cv = PurgedKFold(n_splits=3, purge=1)
    with pytest.raises(ValueError, match="exceeds"):
        list(cv.split(_dates(2)))
    with pytest.raises(ValueError, match="sorted"):
        list(cv.split(dates[::-1]))
    with pytest.raises(ValueError, match="unique"):
        list(cv.split(pd.DatetimeIndex([dates[0], dates[0], dates[1]])))
    with pytest.raises(ValueError, match="horizon"):
        list(cv.split(dates, horizon=0))
    with pytest.raises(ValueError, match="Index"):
        list(cv.split(np.arange(10)))  # type: ignore[arg-type]


def test_get_n_splits() -> None:
    assert PurgedKFold(n_splits=7, purge=0).get_n_splits() == 7
