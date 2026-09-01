# Benchmarks and reference prototype

Scripts backing the measured claims in [the design proposal](https://github.com/XwK-P/medh5/blob/main/design/medh5-1.0-proposal.md) §5 and in the
[specification](../spec/medh5-1.0.md) §7.0, §14.2, §14.3.

| Script | Produces |
|---|---|
| `bench_encodings.py` | Multi-label voxel encodings on a 160³ / 200-class phantom: size, write time, all-class patch read (spec §7.0) |
| `bench_query.py` | Codec-matched single-class vs all-class patch reads across `per-class` / `layers` / `bitmask` (proposal §5.1) |
| `bench_io.py` | Codec profiles, `int16 + rescale` vs `float32`, foreground-index vs `argwhere` (spec §14.2, §14.3) |
| `reference_writer.py` | A complete MEDH5 1.0 file exercising `core+seg+det+cls+reg+curation+training+longitudinal` — one subject at two timepoints — then validates `/meta` against the JSON Schema and runs the §15 semantic and integrity checks |

```bash
pip install numpy h5py hdf5plugin jsonschema
cd docs/examples
python bench_encodings.py     # ~15 s
python bench_query.py         # ~30 s
python bench_io.py            # ~20 s
python reference_writer.py    # ~10 s, writes case_0001.medh5
```

Recorded numbers were taken on macOS (Apple silicon), Python 3.12, h5py 3.16, hdf5plugin 6.0, local
SSD, medians over 10–50 repetitions. Absolute timings are hardware-dependent; the ratios between
encodings are not.

`reference_writer.py` is the executable proof that the specification is self-consistent: it follows
the spec literally and its output passes JSON Schema validation, cross-reference checks (E1xx–E6xx),
per-object digests and `content_id` (E7xx), plus reader-side round-trips for the affine, the box↔slice
convention, instance mask decoding and lossless `layers ↔ bitmask` transcoding.

The sample it writes is longitudinal: baseline CT + PET sharing one frame of reference, a follow-up CT
on its own grid with shorter z coverage and its own frame, organ and lesion annotations at both
visits, a RECIST response label spanning them, and the registration relating the two. It therefore
also exercises the timepoint rules (§3.7 — E106/E107/E108), cross-timepoint instance tracking (§7.4 —
persisted / resolved / new), change labels (§9 — E409) and the longitudinal warnings W909–W911.

> **Note.** Reading a Blosc2-compressed MEDH5 file requires `import hdf5plugin` before the read, or
> HDF5 raises a confusing plugin-path error rather than a missing-filter error. This is the reason the
> spec defines the `portable` codec profile (§14.2).
