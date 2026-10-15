# The cohort manifest

What `medh5 dataset index` and `Manifest.save` write: one metadata-only scan of
a cohort, which splitting, statistics and the
[cohort checks](cohort-checks.md) read without opening a file. The sample
specification calls the dataset manifest authoritative for splits (1.0 §12.3)
and gives it no format; this page is that format's contract, versioned on its
own as `medh5.dataset/1`.

```json
{
  "schema": "medh5.dataset/1",
  "format": "1.0",
  "generator": "medh5 2.1.0",
  "root": ".",
  "sha256": "4f21c8ab90e1…",
  "samples": 412,
  "entries": [{"path": "site-a/case-0001.medh5", "sample_id": "case-0001", "…": "…"}]
}
```

`format` is the sample format version of the writer, `sha256` the digest of the
cohort's membership (`sample_id`, `subject_id` and `group_id` of every entry,
and nothing else), and each entry one sample's metadata
([the fields](../guides/cohorts.md#index-the-directory)).

## Paths

In a `medh5.dataset/1` manifest, `root` and each entry's `path` are absolute or
relative to the directory holding the manifest; a reader resolves a relative
path against that directory, never against its working directory. A writer
writes a path relative only when it lies beneath that directory, absolute
otherwise.

A relative path never holds `..`: a lexical `..` and the system's disagree
across a symbolic link. Relative paths are written with `/`, which every
platform reads.

In memory, medh5 keeps a relative path relative to the working directory, as
the caller named it: `Manifest.load` joins each stored path onto the manifest's
directory, and `Manifest.save` re-bases it onto the directory it writes to. A
path given absolute stays absolute unless `paths="relative"` is passed;
`paths="absolute"` (or `medh5 dataset index --absolute`) writes every path
absolute. No path is covered by `sha256`, so re-basing never changes it.
`medh5.dataset.manifest.PathStyle` names the three choices, and
`medh5.dataset.manifest.SCHEMA` is the `schema` every save writes.
`medh5.dataset.scan(root)` indexes a directory into a manifest,
`medh5.dataset.manifest.load(path)` reads a saved one, and
`medh5.dataset.split.load(path)` reads a split `medh5 dataset split -o` wrote.

## Manifests written before 2.1

A manifest with no `schema` was written by medh5 2.0 or earlier, with paths as
its scan named them: relative to the directory the scan ran in. medh5 2.1 reads
it that way, against the working directory, and raises a `FutureWarning` (the
native command line prints `warning: …`) where that reading and a
`medh5.dataset/1` reading differ, that is, where some path is relative and the
manifest is not in the working directory. `Manifest.load(path, base=…)` names
the directory its paths are relative to, and
[`medh5 dataset upgrade`](cli.md#medh5-dataset-upgrade) rewrites it as a
`medh5.dataset/1` manifest. medh5 3.0 refuses such a manifest unless `base` is
given.

A manifest naming any other `schema` is refused.

## Related

- [Build and split a cohort](../guides/cohorts.md) — the task the manifest serves.
- [Cohort check codes](cohort-checks.md) — what `medh5 dataset check` reports.
