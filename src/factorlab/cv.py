"""
factorlab.cv — Purged cross-validation for overlapping financial labels.

Implements the Purged K-Fold scheme of López de Prado (2018), *Advances in
Financial Machine Learning*, ch. 7.  A label observed at date ``t`` spans
``[t, t + horizon - 1]``.  If such a label sits in the training set and
overlaps the test window, the model sees the future of the test fold.  Purging
removes every training date whose label overlaps the test fold; the embargo
additionally removes dates immediately after the test fold to break serial
correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedKFold:
    """
    Purged K-Fold splitter over a sorted date index.

    Folds are contiguous blocks of dates.  For a test block ``[s, e)``:

    * training dates before the block end at ``s - purge - embargo - horizon``
      (inclusive), so their labels end at least ``purge + embargo`` days
      before ``s``;
    * training dates after the block start at ``e + purge + embargo + horizon - 1``,
      so their labels start at least ``purge + embargo`` days after the block.

    Args:
        n_splits: Number of folds (>= 2).
        purge: Minimum number of days between a training label and the test
            window (>= 0).
        embargo: Extra days removed after the test window (>= 0).
    """

    n_splits: int
    purge: int
    embargo: int = 0

    def __post_init__(self) -> None:
        for name in ("n_splits", "purge", "embargo"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.purge < 0:
            raise ValueError("purge must be >= 0")
        if self.embargo < 0:
            raise ValueError("embargo must be >= 0")

    def split(
        self,
        dates: pd.DatetimeIndex | pd.Index,
        horizon: int = 1,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield ``(train_idx, test_idx)`` integer positional arrays.

        Args:
            dates: Sorted, unique date index (length ``n``).
            horizon: Label horizon in days (>= 1); labels span
                ``[t, t + horizon - 1]``.

        Yields:
            ``(train_idx, test_idx)`` with sorted positional indices.

        Raises:
            ValueError: If ``dates`` is not a sorted unique index, if
                ``horizon < 1``, or if there are too few dates.
        """
        if not isinstance(dates, pd.Index):
            raise ValueError("dates must be a pandas Index")
        if len(dates) < 2:
            raise ValueError("dates must contain at least two entries")
        if not dates.is_unique:
            raise ValueError("dates must be unique")
        if not dates.is_monotonic_increasing:
            raise ValueError("dates must be sorted ascending")
        if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool):
            raise ValueError("horizon must be an integer")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        n = len(dates)
        if self.n_splits > n:
            raise ValueError(
                f"n_splits={self.n_splits} exceeds the number of dates ({n})"
            )

        fold_size = n // self.n_splits
        if fold_size < 1:
            raise ValueError("Too many splits for the available dates")

        # Distance (in dates) kept clear on each side of the test block.
        gap = self.purge + self.embargo + int(horizon)

        for fold in range(self.n_splits):
            start = fold * fold_size
            end = n if fold == self.n_splits - 1 else (fold + 1) * fold_size
            test_idx = np.arange(start, end, dtype=int)

            before_end = start - gap
            after_start = end + gap
            parts = []
            if before_end > 0:
                parts.append(np.arange(0, before_end, dtype=int))
            if after_start < n:
                parts.append(np.arange(after_start, n, dtype=int))
            train_idx = (
                np.concatenate(parts) if parts else np.empty(0, dtype=int)
            )
            yield train_idx, test_idx

    def get_n_splits(self, *args: object, **kwargs: object) -> int:
        """Number of splits (sklearn-compatible signature)."""
        return int(self.n_splits)


def purged_cv_splits(
    n_samples: int,
    n_splits: int,
    purge: int,
    embargo: int = 0,
    horizon: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Materialize all purged splits for a synthetic business-day index.

    Args:
        n_samples: Number of dates (>= 2).
        n_splits: Number of folds (>= 2).
        purge: Purge days.
        embargo: Embargo days.
        horizon: Label horizon.

    Returns:
        List of ``(train_idx, test_idx)`` tuples.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2")
    cv = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo)
    dates = pd.date_range("2020-01-01", periods=int(n_samples), freq="B")
    return list(cv.split(dates, horizon))
