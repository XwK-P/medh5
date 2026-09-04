"""Publishing the corpus, and scoring an implementation that is not this one.

``run_corpus`` answers "does *this* validator agree with the spec".  The
question a format has to answer at 1.0 is different: **does yours?**  So the
corpus ships as a directory anyone can download --- files, expected diagnostics,
the code table, the JSON Schema and checksums --- and any implementation, in any
language, is scored by handing back a list of what it reported per file.

The submitted format is deliberately the smallest thing that can be checked::

    [{"file": "core-minimal.medh5", "errors": ["E101"], "warnings": []}, ...]

``medh5 validate --json`` already emits a superset of it (``path`` plus a
``diagnostics`` list), so the reference implementation scores itself through the
same door as everybody else --- which is the only way the door stays honest.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import medh5
from medh5.conformance.corpus import CASES, Case, CaseResult, build_corpus
from medh5.errors import CODES, MEDH5ValidationError

SCHEMA = "medh5-sample-1.0.schema.json"
CHECKSUMS = "SHA256SUMS"


def publish(
    outdir: str | os.PathLike[str], *, names: Sequence[str] | None = None
) -> Path:
    """Write the distributable suite and return its directory.

    Everything an independent implementation needs is in one place: it should
    not have to install this package to be measured against the spec.
    """
    root = Path(os.fspath(outdir))
    build_corpus(root, names=names)
    selected = [c for c in CASES if names is None or c.name in set(names)]

    (root / "codes.json").write_text(
        json.dumps(
            {
                "format": medh5.FORMAT_VERSION,
                "codes": [
                    {
                        "code": code.code,
                        "severity": code.severity,
                        "domain": code.domain,
                        "summary": code.summary,
                    }
                    for code in CODES.values()
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    schema = Path(medh5.__file__).parent / "schemas" / SCHEMA
    (root / SCHEMA).write_text(schema.read_text(encoding="utf-8"))
    (root / "README.md").write_text(_readme(selected), encoding="utf-8")
    _write_checksums(root)
    return root


def _write_checksums(root: Path) -> Path:
    """Checksum every published file except the checksum file itself."""
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == CHECKSUMS:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    target = root / CHECKSUMS
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def check_checksums(root: str | os.PathLike[str]) -> tuple[str, ...]:
    """Names of published files whose bytes no longer match ``SHA256SUMS``."""
    directory = Path(os.fspath(root))
    bad: list[str] = []
    for line in (directory / CHECKSUMS).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        path = directory / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return tuple(bad)


def load_manifest(root: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(os.fspath(root)) / "expected.json"
    if not path.exists():
        raise MEDH5ValidationError(
            f"{path} not found --- build the suite first with "
            "`medh5 conformance publish`"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _codes(entry: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Errors and warnings from either accepted submission shape.

    A foreign implementation sends ``errors``/``warnings`` lists.  ``medh5
    validate --json`` sends ``diagnostics`` with a severity on each.  Both are
    read here so nobody has to reshape a report to be scored.
    """
    if "diagnostics" in entry:
        errors, warnings = set(), set()
        for diagnostic in entry["diagnostics"] or ():
            code = str(diagnostic.get("code", ""))
            if not code:
                continue
            if str(diagnostic.get("severity", "error")) == "warning":
                warnings.add(code)
            else:
                errors.add(code)
        return errors, warnings
    return (
        {str(c) for c in entry.get("errors", ()) or ()},
        {str(c) for c in entry.get("warnings", ()) or ()},
    )


def _name(entry: Mapping[str, Any]) -> str:
    """The case a submitted result is about, keyed by file name."""
    raw = entry.get("file") or entry.get("path") or entry.get("name") or ""
    return Path(str(raw)).name


def score(
    root: str | os.PathLike[str],
    submitted: Iterable[Mapping[str, Any]],
) -> list[CaseResult]:
    """Score a foreign validator's results against the published expectations.

    A case with no submitted result is a failure, not a skip: silence about a
    file you were given is the same as failing to diagnose it.
    """
    manifest = load_manifest(root)
    by_file = {_name(entry): entry for entry in submitted}
    by_name = {case.name: case for case in CASES}
    results: list[CaseResult] = []
    for record in manifest["cases"]:
        case = by_name.get(record["name"]) or _case_from(record)
        path = str(Path(os.fspath(root)) / record["file"])
        result = CaseResult(case=case, path=path)
        entry = by_file.get(record["file"])
        if entry is None:
            result.error = "no result submitted for this case"
            results.append(result)
            continue
        errors, warnings = _codes(entry)
        result.got_errors = tuple(sorted(errors))
        result.got_warnings = tuple(sorted(warnings))
        expected = set(record["expect_errors"]) | set(record["expect_warnings"])
        got = errors | warnings
        result.missing = tuple(sorted(expected - got))
        result.unexpected = tuple(sorted(got - expected))
        results.append(result)
    return results


def _case_from(record: Mapping[str, Any]) -> Case:
    """A Case standing in for a manifest entry this build does not know.

    Scoring a suite published by a newer medh5 should report on its cases, not
    drop them.
    """
    return Case(
        name=str(record["name"]),
        description=str(record.get("description", "")),
        clause=str(record.get("clause", "")),
        build=lambda _: None,
        level=record.get("level", "semantic"),
        errors=tuple(record.get("expect_errors", ())),
        warnings=tuple(record.get("expect_warnings", ())),
        suffix=str(record.get("file_suffix", ".medh5")),
        mutated=bool(record.get("mutated", False)),
    )


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    failures = [r for r in results if not r.ok]
    return {
        "cases": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "ok": not failures,
        "failures": [r.to_json() for r in failures],
    }


def _readme(cases: Sequence[Case]) -> str:
    valid = sum(1 for c in cases if c.valid)
    shards = sum(1 for c in cases if c.suffix == ".medh5c")
    kinds = f"the cases: {len(cases) - shards} samples and {shards} collections"
    covered = sorted({code for c in cases for code in (*c.errors, *c.warnings)})
    return f"""\
# MEDH5 {medh5.FORMAT_VERSION} conformance suite

Generated by medh5 {medh5.__version__}. {len(cases)} cases: {valid} valid files a
conforming implementation must accept, {len(cases) - valid} invalid ones it must
reject with specific diagnostic codes. {len(covered)} of the codes in the
specification's §15.2 table appear here.

## What is in this directory

| File | What it is |
|---|---|
| `*.medh5`, `*.medh5c` | {kinds} |
| `expected.json` | per case, the exact codes a conforming validator must emit |
| `codes.json` | the §15.2 diagnostic code table as data |
| `{SCHEMA}` | the JSON Schema for the `/meta` document |
| `{CHECKSUMS}` | sha256 of every file above |

## Running it

Validate every case **at the level its manifest entry declares**, and write one
JSON array:

```json
[
  {{"file": "core-minimal.medh5", "errors": [], "warnings": []}},
  {{"file": "E102-not-orthonormal.medh5", "errors": ["E102"], "warnings": []}}
]
```

Then score it:

```
medh5 conformance score . results.json
```

`medh5 validate --json` emits a superset of that shape (a `diagnostics` list
with a severity on each), so the reference implementation is scored through
exactly the same door as everybody else:

```python
import json, subprocess
manifest = json.load(open("expected.json"))
results = []
for case in manifest["cases"]:
    out = subprocess.run(
        ["medh5", "validate", case["file"], "--level", case["level"], "--json"],
        capture_output=True, text=True,
    ).stdout
    report = json.loads(out)[0]
    results.append({{"file": case["file"], "diagnostics": report["diagnostics"]}})
json.dump(results, open("results.json", "w"))
```

## How it is scored

For each case, the set of codes you report must **equal** the expected set: a
missing code is a defect you failed to catch, an extra code is a valid file you
rejected. Both fail. A case you report nothing about fails too --- silence about
a file you were handed is not a pass.

Diagnostic *messages* are yours to write; only the codes are normative.

## Three things worth knowing before you start

**Validate at the declared `level`, not deeper.** `structural` < `semantic` <
`integrity`. Shallower misses the defect the case exists to test. *Deeper is
not safe either*: most invalid cases were made by editing a valid file, so
their stored digests cover the pre-edit bytes and an integrity pass adds a
`content_id` mismatch the case never claimed. Those cases are marked
`"mutated": true`, and the mismatch is an artifact of how they are built, not
something to diagnose.

**A `.medh5c` case is a collection** (spec §2.1) --- it contains samples rather
than being one. `"file_suffix"` in the manifest tells you which.

**Verify the bytes first.** `{CHECKSUMS}` covers every published file; `medh5
conformance score` warns when a case has drifted, because a score over files
that are not the published files is not a score.

The specification is `docs/spec/medh5-{medh5.FORMAT_VERSION}.md` in the medh5
repository, and every case names the clause it tests in `expected.json`.
"""


__all__ = [
    "CHECKSUMS",
    "check_checksums",
    "load_manifest",
    "publish",
    "score",
    "summarize",
]
