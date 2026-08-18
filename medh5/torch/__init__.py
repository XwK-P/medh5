"""PyTorch integration (implementation plan §2.3).

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

``worker_init_fn`` is not optional: without it a forked worker inherits the
parent's HDF5 handles and returns corrupt reads (§14.4).
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
from medh5.torch.collate import collate, stack_images
from medh5.torch.datasets import (
    ALIGNMENTS,
    LABEL_FORMATS,
    GridPatchDataset,
    PairedPatchDataset,
    PatchDataset,
    VolumeDataset,
)
from medh5.torch.handles import CACHE, HandleCache, open_cached, worker_init_fn

__all__ = [
    "ALIGNMENTS",
    "AVAILABLE",
    "CACHE",
    "LABEL_FORMATS",
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
    "grid_patches",
    "open_cached",
    "require_torch",
    "stack_images",
    "worker_init_fn",
]
