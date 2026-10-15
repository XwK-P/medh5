# Pack small samples into collections

One sample per file is the default, and the right one for volumes: a file is the unit a split
assigns, a digest covers and `ln -s` links. When the samples are small (2-D radiographs, patches,
cell crops) and there are a hundred thousand of them, the files themselves become the problem. A
**collection** (`.medh5c`) holds many sample roots in one HDF5 file, each exactly as it would be on
its own.

## Pack and unpack

```python
import medh5

medh5.pack(["case.medh5", "case2.medh5"], "pair.medh5c", keys=["case_0001", "case_0002"])
```

```bash
medh5 pack case.medh5 case2.medh5 --key case_0001 --key case_0002 -o pair.medh5c
medh5 ls pair.medh5c
```

Keys default to the file stems. Packing copies chunks as they are stored: nothing is decompressed
or recompressed, so each member keeps the `content_id` its file had. The reverse is just as exact
(`shard.medh5c` holds `case.medh5` as `case_0001`):

```python
restored = medh5.unpack("shard.medh5c", "restored/", keys=["case_0002"])
one = medh5.collection.extract("shard.medh5c", "case_0001", "member.medh5")

with medh5.open("case.medh5") as original, medh5.open(one) as extracted:
    assert extracted.content_id == original.content_id
```

## Read a member

A collection is a read-only mapping from key to sample, and each member is an ordinary `Sample`:
every reader, validator and loader works on it unchanged.

```python
with medh5.open_collection("shard.medh5c") as shard:
    assert list(shard) == ["case_0001", "case_0002"]
    member = shard["case_0001"]
    ct = member.images["CT"].read((slice(0, 8),) * 3)
    sizes = {key: sample.images["CT"].shape for key, sample in shard.items()}
```

When code is handed a path and does not know which kind it is, `open_any` opens either, and
resolves a key inside a collection:

```python
from medh5.collection import open_any

with open_any("shard.medh5c", key="case_0002") as sample:
    assert sample.kind == "sample"
with open_any("shard.medh5c") as opened:
    assert opened.kind == "collection"
```

A collection cannot be amended in place: `medh5.amend` refuses one and says so. Extract the
member, amend the file, and pack again.

## Members in a cohort

A manifest scanned from a directory lists each member as its own entry, with the shard's path and
the member's `key`. A split assigns members, never shards, so packing changes nothing about who
lands in which partition: two members of one subject land together, as two files would.

```python
import shutil
from pathlib import Path

from medh5.dataset import compute_stats, make_splits
from medh5.dataset.manifest import scan

Path("packed").mkdir()
shutil.copy("shard.medh5c", "packed/")
manifest, problems = scan("packed")
assert [(Path(e.path).name, e.key) for e in manifest] == [
    ("shard.medh5c", "case_0001"),
    ("shard.medh5c", "case_0002"),
]

split = make_splits(manifest, group_by="subject_id", ratios={"train": 0.5, "val": 0.5})
train = split.paths("train") or split.paths("val")   # one subject: one partition holds both
assert all("shard.medh5c::" in p for p in train)
stats = compute_stats(train, images=["CT"])
```

A member is named `path::key` wherever a path is expected to name a sample: in a split's paths,
in `compute_stats`, and in the MONAI reader (`MedH5Reader`).

## Verify every member

Each member carries its own digests and `content_id`. Check them member by member:

```python
with medh5.open_collection("shard.medh5c") as shard:
    failed = [key for key, sample in shard.items() if not sample.verify().ok]
assert failed == []
```

```bash
medh5 validate shard.medh5c
medh5 verify shard.medh5c --key case_0001
```

`validate` takes a collection whole and checks every member. `verify` reports on one sample, so
it asks for the member with `--key`.

## How big a shard

Nothing in the format limits a shard, so the limits are operational:

- **Change.** A shard is written whole by `pack`. Changing one member means extracting it,
  amending it and packing again, so keep members that change often out of large shards.
- **Moving.** A shard is the unit you copy, checksum and archive, so one shard for a whole dataset
  makes every transfer all or nothing.

A volume of hundreds of megabytes gains nothing from a collection: its own file is already large
enough that the per-file cost does not show.

## Related

- [Storage › Collections](../reference/storage.md#collections): the container rules and the
  commands.
- [Build and split a cohort](cohorts.md): manifests, splits and statistics over files and members.
- [Specification §2.2](../spec/medh5-1.0.md): the normative statement.
