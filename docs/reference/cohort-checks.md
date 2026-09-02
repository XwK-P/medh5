# Cohort check codes

What `medh5 dataset check` reports. These describe a **cohort** — a set of files
considered together — not a file.

That is why they are a separate code space from the
[diagnostic codes](diagnostic-codes.md), and deliberately so: every file in a
cohort can be individually valid while the cohort is unusable, because two sites
used different label sets, or a subject appears in both train and test. No
per-file validator can see either problem, so no per-file code exists for them.

The codes are tooling-level rather than normative: the specification does not
require an implementation to emit them, and a minor version may add to them
freely. The table is generated from
[`medh5/dataset/check.py`](https://github.com/XwK-P/medh5/blob/main/medh5/dataset/check.py)
at build time.

```bash
medh5 dataset check cohort.json --deep
```

<!--@cohort-codes-->

## Related

- [Build and split a cohort](../guides/cohorts.md) — the task these checks close.
- [Diagnostic codes](diagnostic-codes.md) — the per-file `E`/`W` space.
- [`medh5 dataset check`](cli.md) — flags and JSON output.
