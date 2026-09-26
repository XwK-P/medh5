"""Keeping one file's items together (spec §14.4).

``DataLoader(shuffle=True)`` permutes *items*, and a :class:`PatchDataset` has
``samples_per_volume`` of them per file.  Scattered over an epoch, consecutive
items almost never share a file, so the per-worker handle cache (32 samples by
default) churns: at 100 files and ``shuffle=True``, 76 % of items opened their
file again, and an open is most of the cost of a small patch.

:class:`FileGroupedSampler` shuffles **files** and yields each file's items
together, so an epoch opens each file once per worker that reads it.  The
order is a function of ``(seed, epoch)``: pass it as ``DataLoader(sampler=...)``
in place of ``shuffle=True``, and the dataset's own ``set_epoch`` moves it on.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from medh5.errors import MEDH5ValidationError
from medh5.torch._compat import require_torch, sampler_base

_SamplerBase = sampler_base()


class FileGroupedSampler(_SamplerBase):  # type: ignore[misc,valid-type]
    """Shuffle files, and yield every item of a file before the next file.

    *dataset* is any of this package's datasets: each says which items read
    which file (``file_groups()``).  Items within a file are shuffled too, so
    a batch still mixes the patches it draws.

    The epoch comes from :meth:`set_epoch` when it is called, else from the
    dataset's ``epoch`` (``PatchDataset.set_epoch`` moves both), else from the
    number of epochs this sampler has run.  With a ``DataLoader`` of several
    workers, choose ``samples_per_volume`` a multiple of ``batch_size``: a
    batch is read by one worker, so a file whose items straddle two batches is
    opened by both.
    """

    def __init__(self, dataset: Any, *, shuffle: bool = True, seed: int = 0) -> None:
        require_torch()
        groups = getattr(dataset, "file_groups", None)
        if groups is None:
            raise MEDH5ValidationError(
                f"{type(dataset).__name__} does not say which items read which "
                "file (`file_groups()`), so its items cannot be grouped by file"
            )
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._groups: tuple[tuple[int, ...], ...] = tuple(
            tuple(int(i) for i in group) for group in groups()
        )
        self._epoch: int | None = None
        self._iterations = 0

    def set_epoch(self, epoch: int) -> None:
        """Fix the epoch the next iteration orders for."""
        self._epoch = int(epoch)

    @property
    def groups(self) -> Sequence[Sequence[int]]:
        """Item indices per file, in file order."""
        return self._groups

    def _current_epoch(self) -> int:
        if self._epoch is not None:
            return self._epoch
        epoch = getattr(self.dataset, "epoch", None)
        if epoch is not None:
            return int(epoch)
        return self._iterations

    def __len__(self) -> int:
        return sum(len(group) for group in self._groups)

    def __iter__(self) -> Iterator[int]:
        epoch = self._current_epoch()
        self._iterations += 1
        if not self.shuffle:
            for group in self._groups:
                yield from group
            return
        rng = np.random.default_rng((self.seed, epoch))
        for position in rng.permutation(len(self._groups)):
            items = list(self._groups[int(position)])
            rng.shuffle(items)
            yield from items


__all__ = ["FileGroupedSampler"]
