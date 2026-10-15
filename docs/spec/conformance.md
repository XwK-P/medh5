# Conformance

A format that cannot be checked is a convention. The conformance suite is what
makes "conforming MEDH5 file" a claim somebody else can test.

## The corpus

171 cases, each a file plus the **exact set of diagnostic codes** a conforming
validator must emit for it. 57 are valid files an implementation must accept,
38 of them with specific warnings; 114 are invalid ones it must reject with
specific errors. Between them they hold at least one case for every code in
the specification's tables: 1.0 §15.2, and the codes [1.1](medh5-1.1.md) §11.2
adds.

36 of the cases are the `clinical` profile's (format 1.1): the worked example of
1.1 §9.3 — two imaging visits, an intervening lab, a delayed report and its
revision, a follow-up assessment — a one-visit history, time uncertainty,
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
171/171 cases pass
```

A test in this repository asserts that the specification's tables (1.0 §15.2
and 1.1 §11.2) and the code registry list the same codes, so neither can gain a
code the other lacks.

## Publishing it

```
$ medh5 conformance publish suite/
wrote the suite to suite/: 171 cases, see suite/README.md
```

| File | |
|---|---|
| `*.medh5`, `*.medh5c` | the cases: 165 samples and six collections |
| `expected.json` | per case: the clause, the level, and the expected codes |
| `codes.json` | the diagnostic code table as data (1.0 §15.2 and 1.1 §11.2) |
| `medh5-sample-1.0.schema.json` | the JSON Schema for `/meta` |
| `medh5-clinical-1.schema.json` | the JSON Schema for `clinical/meta` and clinical records (1.1) |
| `medh5-task-1.schema.json`, `medh5-cache-1.schema.json` | the [task and cache contract](task-cache-1.md)'s schemas — published here for convenience, not scored by the corpus |
| `companion/` | the task-and-cache fixtures: task manifests over the suite's clinical samples, feature caches under `caches/`, and `expected.json` (below) |
| `SHA256SUMS` | over every file above |
| `README.md` | the contract, generated with the suite |

Everything an implementer needs is in that directory. Being measured against
the spec does not require installing this package.

**The suite is reproducible.** Built twice, by the same release, it is the
same bytes: no file records when it was written. Every sample's `created` is
`2000-01-01T00:00:00Z`, and no group carries HDF5's creation and
modification times. So a mirror checks a copy of the suite by publishing it
again with the release that made it and comparing the two `SHA256SUMS`, with
no need to trust the copy.

## The task-and-cache fixtures

The corpus scores a format validator. `companion/` scores an implementation of
the separately versioned [task and cache contract](task-cache-1.md): fifteen
task manifests over the suite's own clinical samples, each pinned to the
content those samples have in the suite. `companion/expected.json` gives, for
each, the level it is checked at — `validate`, the manifest alone with no
file opened, or `preflight`, its sources opened too — and the exact set of
`T` codes checking it must find: one invalid manifest per code from `T101` to
`T306`. For the two valid ones it also gives each row's status, the event
versions strict selection admits at its cutoff, the image filling its slot and
its target label, and, where a row states `inputs`, the eligible annotations
the slot may read — so two implementations of prospective selection can be
compared on more than their error codes.

```
$ medh5 task preflight suite/companion/valid-two-subjects.task.json --json
```

`companion/caches/` holds nine feature caches over the same samples: a valid
cache of each level, event and patient, and one for each code from `T401` to
`T407`. Each is listed in `companion/expected.json` at the level `cache`, with
the findings validating it must report, and with the task it is validated
against where a finding is about the task or the cache is patient-level. A
cache validated with its task resolves its sources against the task's
directory, and one validated alone against its own (task-cache-1 §7.1).

```
$ medh5 cache validate suite/companion/caches/T406-whole-history.medh5cache --task suite/companion/valid-two-subjects.task.json --json
```

Hand back what checking every fixture reported, one report per fixture naming
it by `file` (or `path`, as `cache validate` prints it), and score it with
`--companion`. A report's `findings` must equal the fixture's, its `ok` must
agree with them, and for a fixture that states its rows, each row must match
— a row reported that the fixture does not state fails it, as a row left out
does.

```
$ medh5 conformance score suite/ reports.json --companion
```

## Running it against your implementation

Validate every case **at the level its manifest entry declares**, and hand back
one JSON array:

```json
[
  {"file": "core-minimal.medh5", "errors": [], "warnings": []},
  {"file": "E102-non-orthonormal.medh5", "errors": ["E102"], "warnings": []}
]
```

```
$ medh5 conformance score suite/ results.json
171/171 cases pass
```

The scorer also takes the shape `medh5 validate --json` reports a file in: a
`diagnostics` list, each entry with its `code` and `severity`, in place of
`errors` and `warnings` (and `path` in place of `file`). So the reference
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

For each case, the errors you report must **equal** the expected errors, and
the warnings the expected warnings. A missing code is a defect you failed to
catch; an extra code is a valid file you rejected. Both fail.

Severity is scored. An expected error you report as a warning is missing as an
error, and reported as a warning nobody expected: a validator that reports only
warnings has accepted every file. Severity is the code table's — an `E` code is
an error and a `W` code a warning — whatever a level such as `strict` promotes.

A case you report nothing about fails too — silence about a file you were
handed is not a pass.

Diagnostic *messages* are yours to write. Only the codes are normative.

## Scoring an implementation of an earlier version

Every case in `expected.json` states the format version it is addressed to,
as `format_version`: the latest version among the codes it expects, and `1.1`
for a clause of 1.1. That is not always what its file declares — the `E009`
and `E011` cases declare 1.0 and expect 1.1's answers. The manifest's
top-level `codes_revision` is the revision of the code table (1.0 §15.2) the
suite was built against.

An implementation of 1.0, or one written against an earlier revision of the
table, has no answer to a case it was never told about. Two bounds hold it to
what it was:

```
$ medh5 conformance score suite/ results.json --format-version 1.0
$ medh5 conformance score suite/ results.json --codes-revision 1
```

A case addressed to a later version, or expecting a code a later revision
allocated, is not scored, and the summary counts it under `skipped`. A code of
a later revision that you report anyway is ignored rather than held against
you. Without a bound, every case is scored.

## Three things to know before you start

**Validate at the declared `level`, not deeper.** `structural` < `semantic` <
`integrity`. Shallower misses the defect the case exists to test. Deeper is not
safe either: most invalid cases were made by editing a valid file, so their
stored digests cover the pre-edit bytes and an integrity pass adds a
`content_id` mismatch the case never claimed. Those cases are marked
`"mutated": true`.

*(That correction came from running it. The README first said deeper was safe;
it is not, and 119 of the cases prove it.)*

**A `.medh5c` case is a collection** (§2.2) — it contains samples rather than
being one. `"file_suffix"` says which.

**Verify the bytes first.** `SHA256SUMS` covers every published file, and
`medh5 conformance score` fails the score when one has drifted, whatever the
cases say, naming it (under `changed` with `--json`). A score over files that
are not the published files is not a score.

## From Python

```python
from medh5.conformance import (
    CASES, publish, score, summarize, load_manifest, check_checksums, run_corpus,
)

publish("suite/")
check_checksums("suite/")           # names of files whose bytes changed

results = score("suite/", submitted)
summarize(results)                  # {"cases": 171, "passed": 171, "ok": True, ...}
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

Stable API, and part of the specification (1.0 §15.2, 1.1 §11.2): a code's
meaning never changes and codes are never reused, so the corpus can assert
exact code sets. All 100 codes are listed in
[Diagnostic codes](../reference/diagnostic-codes.md).

```python
from medh5 import CODES
CODES["E102"].summary     # "`direction` is not orthonormal to 1e-4"
```

A minor version may add codes. It may not change what an existing one means:
1.1 added `E011`, `E801`–`E819`, `W913` and `W914`, and redefined none.

An implementation of 1.0 that reads a 1.1 file is reading a higher minor
version: the clinical cases are not addressed to it. The higher-minor case
(`W913-higher-minor-projection`) is what such an implementation should do with
a newer file than it implements — read the supported projection, report what
it does not know as `W913`, and never amend it.

1.1 also changes what some files claiming 1.0 are expected to report, because
it defines the `clinical` profile that 1.0 did not know. A 1.0 file declaring
`clinical` (`E009-clinical-profile-in-a-1-0-file`) expects `E009`, where a
validator of 1.0 alone reports the unknown profile as `E007`; and clinical
content a file does not declare (`E803-clinical-content-undeclared`) is `E803`,
where such a validator ignores an unknown group. The suite expects the 1.1
answers.
