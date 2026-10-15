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

Under DDP each rank takes one contiguous share of that order, so a file is
split between ranks only where a share ends: at most ``num_replicas - 1``
files an epoch are read by two neighbouring ranks.  ``DistributedSampler``
instead deals items round-robin, which hands every rank a patch of every file.
Every rank's share is the same length, every epoch --- whole files per rank
would give unequal lengths, and unequal lengths hang DDP's collectives.
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

    *dataset* says which items read which file (``file_groups()``), as
    :class:`~medh5.torch.VolumeDataset`, :class:`~medh5.torch.PatchDataset`,
    :class:`~medh5.torch.GridPatchDataset` and
    :class:`~medh5.torch.PairedPatchDataset` and
    :class:`~medh5.torch.ClinicalTaskDataset` (its rows grouped by subject)
    do; one that does not is refused.  Items within a
    file are shuffled too, so a batch still mixes the patches it draws.

    The epoch comes from :meth:`set_epoch` when it is called, else from the
    dataset's ``epoch`` (``PatchDataset.set_epoch`` moves both), else from the
    number of epochs this sampler has run.  With a ``DataLoader`` of several
    workers, choose ``samples_per_volume`` a multiple of ``batch_size``: a
    batch is read by one worker, so a file whose items straddle two batches is
    opened by both.

    **DDP.** ``num_replicas`` and ``rank`` default to
    ``torch.distributed``'s world size and rank when a process group is
    initialised, else 1 and 0.  Every rank orders the epoch as one replica
    would, extends the order cyclically to a multiple of ``num_replicas`` (or,
    with ``drop_last``, cuts it to one), and yields its own contiguous share:
    ``len(sampler)`` items, the same on every rank and every epoch.  Under DDP
    call :meth:`set_epoch` every epoch, on every rank, so the ranks agree on
    the order --- an epoch counted by each sampler drifts apart as soon as one
    rank iterates once more than another.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_last: bool = False,
    ) -> None:
        require_torch()
        world, own = _distributed()
        replicas = world if num_replicas is None else int(num_replicas)
        position = own if rank is None else int(rank)
        if replicas < 1:
            raise ValueError(f"num_replicas must be at least 1, not {replicas}")
        if not 0 <= position < replicas:
            raise ValueError(
                f"rank must be in [0, {replicas - 1}] for {replicas} replica(s), "
                f"not {position}"
            )
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
        self._num_replicas = replicas
        self._rank = position
        self.drop_last = bool(drop_last)

    @property
    def num_replicas(self) -> int:
        """How many ranks share each epoch."""
        return self._num_replicas

    @property
    def rank(self) -> int:
        """Which share of each epoch this sampler yields."""
        return self._rank

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

    def _share(self) -> int:
        """Items per rank: the length of every rank's share, every epoch."""
        total = sum(len(group) for group in self._groups)
        if self.drop_last:
            return total // self._num_replicas
        return -(-total // self._num_replicas)

    def __len__(self) -> int:
        return self._share()

    def _order(self) -> list[int]:
        """One replica's order of the epoch: files shuffled, then the items
        within each, from one generator seeded by ``(seed, epoch)``."""
        epoch = self._current_epoch()
        self._iterations += 1
        if not self.shuffle:
            return [i for group in self._groups for i in group]
        rng = np.random.default_rng((self.seed, epoch))
        order: list[int] = []
        for position in rng.permutation(len(self._groups)):
            items = list(self._groups[int(position)])
            rng.shuffle(items)
            order.extend(items)
        return order

    def __iter__(self) -> Iterator[int]:
        order = self._order()
        share = self._share()
        total = share * self._num_replicas
        if order and len(order) != total:
            # Cut to a multiple of the ranks, or extend cyclically to one.
            order = [order[i % len(order)] for i in range(total)]
        yield from order[self._rank * share : (self._rank + 1) * share]


def _distributed() -> tuple[int, int]:
    """``torch.distributed``'s world size and rank, or ``(1, 0)`` without an
    initialised process group."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size()), int(dist.get_rank())
    return 1, 0


__all__ = ["FileGroupedSampler"]
