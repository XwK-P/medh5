"""PyTorch integration.

``import medh5`` never imports torch; this subpackage does, and says how to
install it when it is absent.

.. code-block:: python

    from torch.utils.data import DataLoader
    from medh5.torch import PatchDataset, PatchSampler, collate, worker_init_fn

    sampler = PatchSampler((96, 96, 96), strategy="balanced", foreground_prob=0.6,
                           foreground_classes=["pancreas", "tumor"],
                           class_weights="inverse_frequency")
    ds = PatchDataset(paths, sampler, images=["CT"],
                      annotations={"organs": ["liver", "pancreas", "tumor"]},
                      label_format="onehot", physical=True)
    loader = DataLoader(ds, batch_size=2, num_workers=8,
                        worker_init_fn=worker_init_fn, collate_fn=collate)

For format 1.1 tasks, :class:`ClinicalTaskDataset` and
:func:`collate_clinical` turn the rows of a ``medh5.task/1`` manifest into
cutoff-aware multimodal batches (see :mod:`medh5.torch.clinical`).

``worker_init_fn`` is recommended rather than required: the handle cache is
PID-keyed and re-checks ownership on every access, so a forked worker abandons
the parent's HDF5 handles on first use rather than reading through or closing
them (§14.4).  The callback does that reset eagerly, at worker start.
"""

from __future__ import annotations

from medh5.sampling import (
    PairReport,
    Patch,
    PatchSampler,
    TimepointPair,
    TimepointPairSampler,
    grid_patches,
)
from medh5.torch._compat import AVAILABLE, require_torch
from medh5.torch.clinical import (
    ClinicalTaskDataset,
    ConceptVocabulary,
    collate_clinical,
)
from medh5.torch.collate import collate, stack_images
from medh5.torch.datasets import (
    ALIGNMENTS,
    LABEL_FORMATS,
    GridPatchDataset,
    PairedPatchDataset,
    PatchDataset,
    VolumeDataset,
)
from medh5.torch.handles import (
    CACHE,
    HandleCache,
    open_cached,
    set_cache_size,
    worker_init_fn,
)
from medh5.torch.samplers import FileGroupedSampler

__all__ = [
    "ALIGNMENTS",
    "AVAILABLE",
    "CACHE",
    "LABEL_FORMATS",
    "ClinicalTaskDataset",
    "ConceptVocabulary",
    "FileGroupedSampler",
    "GridPatchDataset",
    "HandleCache",
    "PairReport",
    "PairedPatchDataset",
    "Patch",
    "PatchDataset",
    "PatchSampler",
    "TimepointPair",
    "TimepointPairSampler",
    "VolumeDataset",
    "collate",
    "collate_clinical",
    "grid_patches",
    "open_cached",
    "require_torch",
    "set_cache_size",
    "stack_images",
    "worker_init_fn",
]
