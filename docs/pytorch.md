# PyTorch integration

Install the `torch` extra:

```bash
pip install "medh5[torch]"
```

## Two dataset classes

### `MEDH5TorchDataset` — eager, full-volume

One `.medh5` file = one sample, loaded in full per `__getitem__`.

```python
from medh5.torch import MEDH5TorchDataset

dataset = MEDH5TorchDataset(paths=["s1.medh5", "s2.medh5", "s3.medh5"])
sample = dataset[0]
sample["images"]["CT"].shape   # torch.Size([128, 256, 256])
sample["label"]
```

Suitable when volumes fit comfortably in memory and you want whole-volume
transforms (e.g. full 3D segmentation on thumbnail-sized inputs).

### `MEDH5PatchDataset` — lazy, patch-based

Uses `PatchSampler` to slice chunk-aligned patches directly from h5py
datasets. The virtual length is `len(paths) * samples_per_volume`.

```python
from medh5.sampling import PatchSampler
from medh5.torch import MEDH5PatchDataset

sampler = PatchSampler(
    patch_size=(96, 96, 96),
    strategy="foreground",      # "uniform" | "foreground" | "balanced"
    foreground_seg="tumor",
    foreground_prob=0.7,
    include_bboxes=False,       # opt-in bbox return (patch-local coords)
)

dataset = MEDH5PatchDataset(
    paths=paths,
    sampler=sampler,
    samples_per_volume=4,
)
```

## Sampling strategies

| Strategy      | Behavior                                                       |
|---------------|----------------------------------------------------------------|
| `"uniform"`   | Uniform over valid patch origins.                              |
| `"foreground"`| Biased toward a named seg mask; cached foreground coord list.  |
| `"balanced"`  | Alternates uniform and foreground draws.                       |

Foreground coordinate lists are cached per file for the lifetime of the
sampler — no repeated `np.argwhere` on every draw.

## Transforms

Pure-numpy, no torch or PIL dependency. Operate on the sample dict
(`{"images": ..., "seg": ..., ...}`).

```python
from medh5.transforms import Compose, Clip, Normalize, ZScore, RandomFlip

transform = Compose([
    Clip(min=-1000, max=1000),
    Normalize(mean={"CT": -200.0}, std={"CT": 350.0}),
    ZScore(),
    RandomFlip(axes=(1, 2), p=0.5),
])

dataset = MEDH5PatchDataset(paths=paths, sampler=sampler, transform=transform)
```

`RandomFlip` flips both voxel data and any bboxes in the sample, and
negates the corresponding column of `meta.spatial.direction` (via
`dataclasses.replace` — the file's cached `SampleMeta` is not mutated) so
physical-space metadata remains consistent with the flipped voxels.

## Handle cache

Both dataset classes share a module-level LRU cache (32 handles by default)
so repeated `__getitem__` calls reuse one open `h5py.File` per path instead
of reopening from scratch.

## `DataLoader(num_workers > 0)`

h5py is not fork-safe, and open `h5py.File` handles cannot be pickled for
`multiprocessing_context="spawn"` (the default on macOS / Windows / Python
3.14+). Two mechanisms keep this working:

1. The handle cache is **PID-scoped**: a forked worker observes the PID
   mismatch and resets to a cold cache instead of inheriting parent h5py
   state.
2. `medh5.torch.worker_init_fn` clears the cache at worker startup as a
   belt-and-braces measure.

Always pass `worker_init_fn` when `num_workers > 0`:

```python
from torch.utils.data import DataLoader
import medh5.torch as mt

loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    worker_init_fn=mt.worker_init_fn,        # <- required
    # multiprocessing_context="spawn",       # optional, default on macOS/Windows
)

for batch in loader:
    ...
```

Without `worker_init_fn`, a forked worker would observe stale h5py state
inherited from the parent process, which can deadlock or silently corrupt
reads. See [h5py's docs](https://docs.h5py.org/en/stable/faq.html#multiprocessing).
