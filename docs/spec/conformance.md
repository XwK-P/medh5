# Conformance

A format that cannot be checked is a convention. The conformance suite is what
makes "conforming MEDH5 file" a claim somebody else can test.

## The corpus

152 cases, each a file plus the **exact set of diagnostic codes** a conforming
validator must emit for it. 49 are valid files an implementation must accept,
32 of them with specific warnings; 103 are invalid ones it must reject with
specific errors. Between them they exercise every code in the specification's
tables --- 1.0 §15.2 and the codes [1.1](medh5-1.1.md) §11.2 adds --- and every
cross-reference clause behind a code.

35 of the cases are the `clinical` profile's (format 1.1): the worked example of
1.1 §9.3 --- two imaging visits, an intervening lab, a delayed report and its
revision, a follow-up assessment --- a one-visit history, time uncertainty,
UTF-8 text, a vocabulary wider than the voxel class space, a collection mixing
1.0 and 1.1 members, a higher minor read as a projection, and one invalid file
at least per clinical code. The cases of 1.0 are unchanged, and a 1.0 file stays a
1.0 file: nothing in the suite asks an implementation of 1.0 to read clinical
content.

Invalid cases are built by mutating a valid one, because the writer refuses to
produce them. That is the point: the writer and the validator are checked
against each other.

```
$ medh5 conformance list
$ medh5 conformance run /tmp/corpus
152/152 cases pass
```

A test in this repository asserts the §15.2 table and the code registry are
identical, so the spec and the implementation cannot drift apart silently.

## Publishing it

```
$ medh5 conformance publish suite/
wrote the suite to suite/: 152 cases, see suite/README.md
```

| File | |
|---|---|
| `*.medh5`, `*.medh5c` | the cases: 146 samples and six collections |
| `expected.json` | per case: the clause, the level, and the expected codes |
| `codes.json` | the diagnostic code table as data (1.0 §15.2 and 1.1 §11.2) |
| `medh5-sample-1.0.schema.json` | the JSON Schema for `/meta` |
| `medh5-clinical-1.schema.json` | the JSON Schema for `clinical/meta` and clinical records (1.1) |
| `medh5-task-1.schema.json`, `medh5-cache-1.schema.json` | the [task and cache contract](task-cache-1.md)'s schemas --- published here for convenience, not scored by the corpus |
| `companion/` | the task-and-cache fixtures: task manifests over the suite's clinical samples, and `expected.json` (below) |
| `SHA256SUMS` | over every file above |
| `README.md` | the contract, generated with the suite |

Everything an implementer needs is in that directory. Being measured against
the spec does not require installing this package.

## The task-and-cache fixtures

The corpus scores a format validator. `companion/` scores an implementation of
the separately versioned [task and cache contract](task-cache-1.md): fourteen
task manifests over the suite's own clinical samples, each pinned to the
content those samples have in the suite. `companion/expected.json` gives, for
each, the level it is checked at --- `validate`, the manifest alone with no
file opened, or `preflight`, its sources opened too --- and the exact set of
`T` codes checking it must find: one invalid manifest per code from `T101` to
`T306`. For the valid one it also gives each row's status, the event versions
strict selection admits at its cutoff, the image filling its slot and its
target label --- so two implementations of prospective selection can be
compared on more than their error codes.

```
$ medh5 task preflight suite/companion/valid-two-subjects.task.json --json
```

## Running it against your implementation

Validate every case **at the level its manifest entry declares**, and hand back
one JSON array:

```json
[
  {"file": "core-minimal.medh5", "errors": [], "warnings": []},
  {"file": "E102-not-orthonormal.medh5", "errors": ["E102"], "warnings": []}
]
```

```
$ medh5 conformance score suite/ results.json
152/152 cases pass
```

`medh5 validate --json` emits a superset of that shape, so the reference
implementation is scored through exactly the same door as everybody else:

<!-- illustrative -->
```python
import json, subprocess

manifest = json.load(open("suite/expected.json"))
results = []
for case in manifest["cases"]:
    out = subprocess.run(
        ["medh5", "validate", f"suite/{case['file']}", "--level", case["level"], "--json"],
        capture_output=True, text=True,
    ).stdout
    report = json.loads(out)[0]
    results.append({"file": case["file"], "diagnostics": report["diagnostics"]})
json.dump(results, open("results.json", "w"))
```

There is a test asserting this path works, because a private door is how a
suite stops being a contract.

## How scoring works

For each case, the set of codes you report must **equal** the expected set. A
missing code is a defect you failed to catch; an extra code is a valid file you
rejected. Both fail.

A case you report nothing about fails too — silence about a file you were
handed is not a pass.

Diagnostic *messages* are yours to write. Only the codes are normative.

## Three things to know before you start

**Validate at the declared `level`, not deeper.** `structural` < `semantic` <
`integrity`. Shallower misses the defect the case exists to test. Deeper is not
safe either: most invalid cases were made by editing a valid file, so their
stored digests cover the pre-edit bytes and an integrity pass adds a
`content_id` mismatch the case never claimed. Those cases are marked
`"mutated": true`.

*(That correction came from running it. The README first said deeper was safe;
it is not, and 109 of the cases prove it.)*

**A `.medh5c` case is a collection** (§2.1) — it contains samples rather than
being one. `"file_suffix"` says which.

**Verify the bytes first.** `SHA256SUMS` covers every published file, and
`medh5 conformance score` warns when a case has drifted. A score over files
that are not the published files is not a score.

## From Python

```python
from medh5.conformance import (
    CASES, publish, score, summarize, load_manifest, check_checksums, run_corpus,
)

publish("suite/")
check_checksums("suite/")           # names of files whose bytes changed

results = score("suite/", submitted)
summarize(results)                  # {"cases": 152, "passed": 152, "ok": True, ...}
```

## Profiles

A file declares which profiles it satisfies, and the validator can hold it to
them:

```
$ medh5 validate case.medh5 --profile det --profile seg
```

The ten profiles and the four validation levels are in
[Profiles and validation levels](../reference/profiles-and-levels.md).

## Diagnostic codes

Stable API, and part of the specification (§15.2): a code's meaning never
changes and codes are never reused, so the corpus can assert exact code sets.
All 93 are listed in [Diagnostic codes](../reference/diagnostic-codes.md).

```python
from medh5 import CODES
CODES["E102"].summary     # "`direction` is not orthonormal to 1e-4"
```

A minor version may add codes. It may not change what an existing one means:
1.1 added `E011`, `E801`–`E819`, `W913` and `W914`, and redefined none.

An implementation of 1.0 that reads a 1.1 file is reading a higher minor
version: the clinical cases are not addressed to it. The higher-minor case
(`W913-higher-minor-projection`) is what such an implementation should do with
a newer file than it implements --- read the supported projection, report what
it does not know as `W913`, and never amend it.
