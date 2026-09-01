# Conformance

A format that cannot be checked is a convention. The conformance suite is what
makes "conforming MEDH5 file" a claim somebody else can test.

## The corpus

103 cases, each a file plus the **exact set of diagnostic codes** a conforming
validator must emit for it. 38 are valid files an implementation must accept;
65 are invalid ones it must reject with specific codes — one per code in the
specification's §15.2 table.

Invalid cases are built by mutating a valid one, because the writer refuses to
produce them. That is the point: the writer and the validator are checked
against each other.

```
$ medh5 conformance list
$ medh5 conformance run /tmp/corpus
103/103 cases pass
```

A test in this repository asserts the §15.2 table and the code registry are
identical, so the spec and the implementation cannot drift apart silently.

## Publishing it

```
$ medh5 conformance publish suite/
wrote the suite to suite/: 103 cases, see suite/README.md
```

| File | |
|---|---|
| `*.medh5`, `*.medh5c` | the cases: 99 samples, and four collections |
| `expected.json` | per case: the clause, the level, and the expected codes |
| `codes.json` | the §15.2 diagnostic code table as data |
| `medh5-sample-1.0.schema.json` | the JSON Schema for `/meta` |
| `SHA256SUMS` | over every file above |
| `README.md` | the contract, generated with the suite |

Everything an implementer needs is in that directory. Being measured against
the spec does not require installing this package.

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
103/103 cases pass
```

`medh5 validate --json` emits a superset of that shape, so the reference
implementation is scored through exactly the same door as everybody else:

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
it is not, and 71 of the cases prove it.)*

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
summarize(results)                  # {"cases": 103, "passed": 103, "ok": True, ...}
```

## Profiles

A file declares which profiles it satisfies, and the validator can hold it to
them:

```
$ medh5 validate case.medh5 --profile det --profile seg
```

The nine profiles and the four validation levels are in
[Profiles and validation levels](reference/profiles-and-levels.md).

## Diagnostic codes

Stable API, and part of the specification (§15.2): a code's meaning never
changes and codes are never reused, so the corpus can assert exact code sets.
All 71 are listed in [Diagnostic codes](reference/diagnostic-codes.md).

```python
from medh5 import CODES
CODES["E102"].summary     # "`direction` is not orthonormal to 1e-4"
```

A minor version may add codes. It may not change what an existing one means.
