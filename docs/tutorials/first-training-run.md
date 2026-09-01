# Your first training run

Take the file you wrote in [the first tutorial](first-sample.md) and feed it to
a PyTorch `DataLoader`. About ten minutes.

```bash
pip install "medh5[torch]"
```

## Build a sampler and a dataset

```python
from torch.utils.data import DataLoader
from medh5.torch import PatchDataset, collate, worker_init_fn
from medh5.sampling import PatchSampler

sampler = PatchSampler((32, 32, 32), strategy="balanced",
                       foreground_classes=["liver", "lesion"])

dataset = PatchDataset(["case_0001.medh5"], sampler,
                       images=["CT"],
                       annotations={"organs": ["liver", "lesion"]},
                       samples_per_volume=8)
```

`strategy="balanced"` draws a foreground centre with probability
`foreground_prob` and a uniform one otherwise. Pure foreground sampling never
shows the model the background it will be evaluated on, which is why `balanced`
is what nearly every segmentation recipe actually uses.

`annotations={"organs": ["liver", "lesion"]}` fixes the channel order. Channel 0
is liver because you asked for liver first — not because of anything in the file.

## Look at one item before you loop

```python
item = dataset[0]
item["images"]["CT"].shape        # (32, 32, 32)
item["label"]["organs"].shape     # (2, 32, 32, 32) — one plane per class
item["meta"]["patch"]["center"]   # [32, 33, 36]
item["meta"]["patch"]["strategy"] # "foreground" or "uniform"
item["meta"]["patch"]["used_index"]
```

`used_index` is the one to look at — but only on a draw whose `strategy` is
`"foreground"`. A uniform draw never consults an index and reports `True`
regardless, so read the two fields together.

A foreground draw reporting `False` means the sampler scanned the volume,
because the file carries no sampling index:

```bash
medh5 index build case_0001.medh5
```

Build it once and that draw becomes a lookup. On this toy volume the difference
is milliseconds; at 512³ it is 312 ms per draw against 0.9.

## Loop

```python
loader = DataLoader(dataset, batch_size=2, num_workers=4,
                    worker_init_fn=worker_init_fn, collate_fn=collate)

for batch in loader:
    batch["images"]["CT"]        # (2, 32, 32, 32)
    batch["label"]["organs"]     # (2, 2, 32, 32, 32)
    batch["meta"]["subject_id"]  # ["DEMO-0001", "DEMO-0001"]
    break
```

`worker_init_fn` drops handles inherited across a `fork`. It is **recommended
but not required for correctness**: the handle cache is PID-keyed and re-checks
ownership on every access, so a forked worker abandons the parent's handles on
first use rather than reading through or closing them. The callback just does
that reset eagerly, at worker start, instead of lazily.

If you need your one `worker_init_fn` slot for seeding or other setup, call it
from your own:

```python
from medh5.torch import worker_init_fn as medh5_worker_init

def init(worker_id):
    medh5_worker_init(worker_id)
    seed_everything(worker_id)
```

`collate` stacks tensors and leaves everything else as lists — which is why
`subject_id` comes back as a list of strings rather than failing inside
`torch.stack`.

## Where to go next

- **[Tune performance](../guides/performance.md)** — the index, chunk sizing, workers.
- **[Partial labels and coverage](../guides/partial-labels.md)** — before you train on a real, partly annotated cohort.
- **[Build and split a cohort](../guides/cohorts.md)** — more than one file.
- **[PyTorch and MONAI](../reference/torch.md)** — every dataset and sampler.
