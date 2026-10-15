"""Stitching patch predictions back into volumes: sliding-window inference.

:class:`~medh5.torch.GridPatchDataset` covers each file with patches;
:class:`GridStitcher` takes the model's predictions for them, batch by batch
in whatever order a ``DataLoader`` delivers them, and hands back each file's
volume as soon as its last patch arrives.  The arithmetic is
:class:`~medh5.sampling.PatchAggregator`'s, shared with code that does not use
torch.

A file's completion is counted from its items' ``meta["cover"]`` --- which of
its *n* patches each one is --- never by matching paths, so a file listed
twice is two volumes, and a patch delivered twice is refused rather than
counted.  A volume is never returned partial: :meth:`GridStitcher.finish`
refuses, naming every file still missing patches, unless ``partial=True``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.sampling import PatchAggregator
from medh5.torch.datasets import GridPatchDataset


def _numpy(predictions: Any) -> npt.NDArray[Any]:
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    return np.asarray(predictions)


def _covers(meta: Any, batch: int) -> list[tuple[int, int, int]]:
    """``(file, index, of)`` per item, from ``meta`` as
    :func:`~medh5.torch.collate` batches it (a mapping of lists) or as a list
    of the items' own ``meta``."""
    if isinstance(meta, Mapping):
        cover = meta.get("cover")
        if not isinstance(cover, Mapping):
            raise MEDH5ValidationError(
                "the batch's meta has no 'cover': stitch the items of a "
                "GridPatchDataset"
            )
        columns = [
            list(map(int, _numpy(cover[key]).tolist()))
            for key in ("file", "index", "of")
        ]
        found = list(zip(*columns, strict=True))
    else:
        found = []
        for item in meta:
            cover = item.get("cover") if isinstance(item, Mapping) else None
            if not isinstance(cover, Mapping):
                raise MEDH5ValidationError(
                    "an item's meta has no 'cover': stitch the items of a "
                    "GridPatchDataset"
                )
            found.append((int(cover["file"]), int(cover["index"]), int(cover["of"])))
    if len(found) != batch:
        raise MEDH5ValidationError(
            f"{batch} prediction(s) and {len(found)} item meta: one each"
        )
    return found


class GridStitcher:
    """Reassemble a :class:`~medh5.torch.GridPatchDataset`'s predictions.

    ``channels`` is the number of channels each prediction carries before its
    spatial axes (``None`` for predictions shaped as the patch); ``mode``,
    ``sigma_scale`` and ``dtype`` are
    :class:`~medh5.sampling.PatchAggregator`'s.  One accumulator is open per
    file that has some but not all of its patches, so memory follows the
    number of files a loader has in flight, not the dataset.
    """

    def __init__(
        self,
        dataset: GridPatchDataset,
        *,
        channels: int | None = None,
        mode: str = "gaussian",
        sigma_scale: float = 0.125,
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        if not isinstance(dataset, GridPatchDataset):
            raise MEDH5ValidationError(
                f"a GridStitcher stitches a GridPatchDataset's cover, not a "
                f"{type(dataset).__name__}"
            )
        self.dataset = dataset
        self.channels = channels
        self.mode = mode
        self.sigma_scale = float(sigma_scale)
        self.dtype = np.dtype(dtype)
        # Refuse a bad configuration now, not at the first batch.
        PatchAggregator((1,), channels=channels, mode=mode, sigma_scale=sigma_scale)
        self._open: dict[int, tuple[PatchAggregator, set[int]]] = {}
        self._done: set[int] = set()

    @property
    def pending(self) -> dict[str, int]:
        """Per file not yet returned, the patches still missing."""
        out: dict[str, int] = {}
        starts = self.dataset._starts
        for file, path in enumerate(self.dataset.paths):
            if file in self._done:
                continue
            end = starts[file + 1] if file + 1 < len(starts) else len(self.dataset)
            seen = self._open.get(file)
            missing = (end - starts[file]) - (0 if seen is None else len(seen[1]))
            out[path] = out.get(path, 0) + missing
        return out

    def add(
        self, predictions: Any, meta: Mapping[str, Any] | Sequence[Mapping[str, Any]]
    ) -> list[tuple[str, npt.NDArray[Any]]]:
        """Add one batch of predictions, ``(B, [channels,] *patch)``, with the
        batch's ``meta``; return ``(path, volume)`` for every file it completes.
        """
        array = _numpy(predictions)
        covers = _covers(meta, len(array))
        completed: list[tuple[str, npt.NDArray[Any]]] = []
        for prediction, (file, k, n) in zip(array, covers, strict=True):
            if not 0 <= file < len(self.dataset.paths):
                raise MEDH5ValidationError(
                    f"cover file {file} is not one of the dataset's "
                    f"{len(self.dataset.paths)}"
                )
            path = self.dataset.paths[file]
            if file in self._done:
                raise MEDH5ValidationError(
                    f"{path}: patch {k} of {n} arrived after the volume was complete"
                )
            start = self.dataset._starts[file]
            _, patch = self.dataset._plan[start + k]
            entry = self._open.get(file)
            if entry is None:
                shape, grid_id = self.dataset._volumes[file]
                aggregator = PatchAggregator(
                    shape,
                    channels=self.channels,
                    mode=self.mode,
                    sigma_scale=self.sigma_scale,
                    dtype=self.dtype,
                    grid_id=grid_id,
                )
                entry = (aggregator, set())
                self._open[file] = entry
            aggregator, seen = entry
            if k in seen:
                raise MEDH5ValidationError(f"{path}: patch {k} of {n} arrived twice")
            aggregator.add(prediction, patch)
            seen.add(k)
            if len(seen) == n:
                del self._open[file]
                self._done.add(file)
                completed.append((path, aggregator.result()))
        return completed

    def finish(self, *, partial: bool = False) -> list[tuple[str, npt.NDArray[Any]]]:
        """End the run: refuse while any file is missing patches, naming
        each; with ``partial=True`` return the volumes of the files begun but
        not completed instead, ``NaN`` where no patch arrived."""
        missing = {path: n for path, n in self.pending.items() if n}
        if missing and not partial:
            listed = "; ".join(f"{path}: {n} missing" for path, n in missing.items())
            raise MEDH5ValidationError(
                f"{len(missing)} file(s) are missing patches, so no volume of "
                f"theirs is complete ({listed}): feed every item of the "
                "dataset, or finish(partial=True) for what arrived"
            )
        out = [
            (self.dataset.paths[file], aggregator.result())
            for file, (aggregator, _) in sorted(self._open.items())
        ]
        for file in self._open:
            self._done.add(file)
        self._open.clear()
        return out


__all__ = ["GridStitcher"]
