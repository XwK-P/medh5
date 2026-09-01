# Diagnostic codes

Every defect `medh5 validate` can report, with the code it reports it under.

Codes are **stable API**. A code's meaning never changes and codes are never
reused, so a pipeline may branch on one and a third-party validator is expected
to emit the same code for the same defect. A minor format version may add codes;
it may not change what an existing one means (spec §16).

The table below is generated from
[`medh5/errors.py`](https://github.com/XwK-P/medh5/blob/main/medh5/errors.py) at
build time — the same table the validator and the conformance corpus read, so it
cannot drift from what the tool actually emits. Its normative statement is
[specification §15.2](../spec/medh5-1.0.md#152-error-codes).

```python
from medh5 import CODES

CODES["E102"].summary     # "`direction` is not orthonormal to 1e-4"
CODES["E102"].domain      # "geometry"
CODES["E102"].severity    # "error"
```

## Reading a code

The first digit is the domain, which is usually enough to know which part of a
file to look at:

| Range | Domain |
|---|---|
| `E0xx` | container |
| `E1xx` | geometry |
| `E2xx` | images |
| `E3xx` | label set |
| `E4xx` | annotations |
| `E5xx` | transforms |
| `E6xx` | curation |
| `E7xx` | integrity |
| `W9xx` | warnings, across every domain |

An **error** means the file does not conform. A **warning** means it conforms
but something is worth knowing — a suspicious value, a derived object that is
stale, a claim nothing corroborates. `--level strict` is what turns warnings
into a non-zero exit; see [profiles and validation levels](profiles-and-levels.md).

Warnings are grouped with the domain they concern rather than in a block of
their own, so everything about geometry is in one place.

<!--@diagnostic-codes-->

## Related

- [Profiles and validation levels](profiles-and-levels.md) — which of these get checked, and when.
- [Cohort check codes](cohort-checks.md) — the `C1xx` codes, which are a *different* space: they describe a cohort, not a file.
- [Conformance suite](../spec/conformance.md) — one corpus case per code on this page.
