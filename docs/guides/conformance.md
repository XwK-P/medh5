# Run the conformance suite

<!-- sequential -->

The conformance suite is how a validator, this one or one written in another language, shows that
it reads MEDH5 as the specification says. Each case is a file and the exact codes a conforming
validator must report for it. This page publishes a small suite, validates it, scores the result,
and scores it again as an implementation of an earlier version would be. What the suite holds and
how a score is decided is [the conformance page](../spec/conformance.md); this page is how to run
it. Its blocks run in order.

## Publish a suite

```python
from medh5.conformance import load_manifest, publish

publish("suite", names=[
    "core-minimal",
    "E102-non-orthonormal",
    "W903-no-deidentification",
    "E009-clinical-profile-in-a-1-0-file",
])
manifest = load_manifest("suite")
first = manifest["cases"][0]
assert (first["file"], first["level"], first["format_version"]) == ("core-minimal.medh5", "semantic", "1.0")
```

`names=` picks cases, which keeps an example small; without it you get all of them. From the
shell:

```
$ medh5 conformance publish suite/
wrote the suite to suite/: 175 cases, see suite/README.md
```

Everything an implementer needs is in that directory, `expected.json` and the code table
included, so being measured against the specification does not require installing this package.

## Validate every case at its level

Validate each case **at the level its entry declares**, and collect one result per case: the file,
and the codes you reported as errors and as warnings.

```python
from medh5.validate import validate_file

submitted = []
for case in manifest["cases"]:
    report = validate_file(f"suite/{case['file']}", level=case["level"])
    submitted.append({
        "file": case["file"],
        "errors": [d.code for d in report.errors],
        "warnings": [d.code for d in report.warnings],
    })
```

Another implementation writes the same list as JSON:

```json
[
  {"file": "core-minimal.medh5", "errors": [], "warnings": []},
  {"file": "E102-non-orthonormal.medh5", "errors": ["E102"], "warnings": []}
]
```

The scorer also takes what `medh5 validate --json` prints: `path` in place of `file`, and a
`diagnostics` list, each entry with its `code` and `severity`, in place of `errors` and
`warnings`. So the reference implementation is scored through the same door as everybody else.

Three things trip implementers up:

- **Validate at the declared level, not deeper.** `structural` < `semantic` < `integrity`.
  Shallower misses the defect the case exists for. Deeper is not safe either: most invalid cases
  were made by editing a valid file, so their stored digests cover the bytes before the edit, and
  an integrity pass adds a `content_id` mismatch the case never claimed. Those cases are marked
  `"mutated": true`; 121 of the cases prove it.
- **A `.medh5c` case is a collection** (§2.2): it contains samples rather than being one.
  `"file_suffix"` says which.
- **Verify the bytes first.** `SHA256SUMS` covers every published file, and a score over files that
  are not the published ones is not a score.

## Score it

```python
from medh5.conformance import check_checksums, score, summarize

assert check_checksums("suite") == ()        # no file differs from SHA256SUMS
results = score("suite", submitted)
summary = summarize(results)
assert (summary["passed"], summary["ok"]) == (4, True)
```

```
$ medh5 conformance score suite/ results.json
175/175 cases pass
```

A case passes when the errors you reported equal its expected errors and the warnings its expected
warnings, severity included. A case you report nothing about fails. `medh5 conformance score` also
fails the whole score when a file's bytes have drifted from `SHA256SUMS`, naming it.

## Score an implementation of an earlier version

An implementation of 1.0 has no answer to a case addressed to 1.1, and one written against the
first revision of the code table none to a code a later revision added. Bound the score to what it
was told about:

```python
as_1_0 = score("suite", submitted, format_version="1.0")
assert [r.case.name for r in as_1_0] == ["core-minimal", "E102-non-orthonormal", "W903-no-deidentification"]
as_revision_1 = score("suite", submitted, codes_revision=1)
```

```
$ medh5 conformance score suite/ results.json --format-version 1.0
$ medh5 conformance score suite/ results.json --codes-revision 1
```

The `E009` case declares 1.0 but is addressed to 1.1, which defines the profile it is about, so
the 1.0 score leaves it out. A code of a later revision that you report anyway is ignored rather
than held against you.

## Score a task-and-cache implementation

The full suite also carries `companion/`: task manifests and feature caches that score an
implementation of the [task and cache contract](../spec/task-cache-1.md). Check each fixture as its
entry in `companion/expected.json` says, collect the reports, and score them with `--companion`:

```
$ medh5 task preflight suite/companion/valid-two-subjects.task.json --json
$ medh5 conformance score suite/ reports.json --companion
```

From Python, `medh5.conformance.score_companion(root, submitted)` scores the same reports.

## Related

- [Conformance](../spec/conformance.md): what the corpus holds, the fixtures, and how scoring is
  decided.
- [Diagnostic codes](../reference/diagnostic-codes.md): every code a case can expect.
- [Check a file before training on it](validate.md): `validate` and `verify` on your own files.
