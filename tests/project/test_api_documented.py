"""Every public name has a reference entry, and the hand-written pages catch up.

`docs/reference/api-index.md` is generated from the installed package by
`.github/scripts/api_index.py`, as `THIRD_PARTY_NOTICES` is from the lock: a
row per name in a public module's `__all__`, with its docstring's first
sentence and the section of the pages that documents it.  At 2.0.2, 470 of the
names were named on no page outside the specification.
`tests/project/undocumented_api.txt` lists the ones still without a section,
and may only shrink: a pull request that adds a public name documents it.  A
test cannot see the base a pull request is against, so CI's `lint` job refuses
a line the pull request adds to the list.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

from tests.helpers import ROOT

# The index covers `medh5.torch`, whose classes are what PyTorch makes them.
pytest.importorskip("torch")


def _api_index() -> ModuleType:
    path = ROOT / ".github" / "scripts" / "api_index.py"
    spec = importlib.util.spec_from_file_location("api_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["api_index"] = module  # dataclasses resolves the module by name
    spec.loader.exec_module(module)
    return module


api_index = _api_index()


@pytest.fixture(scope="module")
def rows() -> list:
    return api_index.rows()


def test_the_api_index_is_current() -> None:
    assert api_index.main(["--check"]) == 0, (
        "docs/reference/api-index.md is stale: run python .github/scripts/api_index.py"
    )


def test_every_public_name_is_documented_or_listed(rows: list) -> None:
    listed = set(api_index.listed())
    missing = sorted(
        row.path for row in rows if row.section is None and row.path not in listed
    )
    assert not missing, (
        "public names no page documents; name each in the section of "
        f"docs/reference/ that covers it: {missing}"
    )


def test_the_undocumented_list_only_shrinks(rows: list) -> None:
    """No listed name is now documented, or gone: either is a line to delete."""
    by_path = {row.path: row for row in rows}
    stale = [
        path
        for path in api_index.listed()
        if path not in by_path or by_path[path].section is not None
    ]
    assert not stale, f"delete from tests/project/undocumented_api.txt: {stale}"
    assert api_index.listed() == sorted(set(api_index.listed()))


def test_every_public_name_has_a_docstring(rows: list) -> None:
    """A constant is exempt: its `__doc__` is its type's."""
    bare = sorted(
        row.path
        for row in rows
        if not row.summary and not row.kind.startswith("constant")
    )
    assert not bare, f"public names without a docstring: {bare}"


def test_a_command_does_not_document_the_function_of_its_name() -> None:
    """`medh5 pack` on the command line's page is the command, not
    `medh5.collection.pack`; the function's entry links the Python pages."""
    found = api_index.sections({"pack", "fix"})
    for name in ("pack", "fix"):
        assert found[name].page not in ("reference/cli.md", "reference/rust.md")


def test_a_qualified_name_documents_that_object_alone(rows: list) -> None:
    """`medh5.dataset.manifest.SCHEMA` on the manifest page documents the
    manifest's constant, not every module's `SCHEMA`."""
    by_path = {row.path: row for row in rows}
    manifest = by_path["medh5.dataset.manifest.SCHEMA"].section
    assert manifest is not None and manifest.page == "reference/cohort-manifest.md"
    for other in ("medh5.task.SCHEMA", "medh5.cache.SCHEMA"):
        section = by_path[other].section
        assert section is None or section.page != "reference/cohort-manifest.md"


def test_a_qualified_constant_does_not_document_its_values_twins(rows: list) -> None:
    """`medh5.errors.CODES_REVISION` is the int 2, which Python shares with
    every other 2: naming it documented `medh5.cli.EXIT_USAGE` and
    `medh5.geometry.MIN_SPATIAL` too."""
    import medh5.cli
    import medh5.errors

    assert medh5.errors.CODES_REVISION is medh5.cli.EXIT_USAGE  # the trap
    by_path = {row.path: row for row in rows}
    revision = by_path["medh5.errors.CODES_REVISION"].section
    assert revision is not None and revision.page == "reference/profiles-and-levels.md"
    for twin in ("medh5.cli.EXIT_USAGE", "medh5.geometry.MIN_SPATIAL"):
        section = by_path[twin].section
        assert section is None or section.page != "reference/profiles-and-levels.md"


def test_a_bare_name_two_objects_share_documents_neither(rows: list) -> None:
    """`load(...)` on the label-set page is `medh5.labels.load`; it says nothing
    of the manifest's or the split's `load`, which their paths document."""
    by_path = {row.path: row for row in rows}
    for path in ("medh5.dataset.manifest.load", "medh5.dataset.split.load"):
        section = by_path[path].section
        assert section is not None and section.page == "reference/cohort-manifest.md"


def test_a_constant_borrows_only_its_own_objects_docstring(rows: list) -> None:
    """A constant with no docstring of its own takes one from the module that
    defines *it*, never from another module's constant of the same name."""
    by_path = {row.path: row for row in rows}
    assert "manifest" not in by_path["medh5.cache.SCHEMA"].summary
    assert "collection" not in by_path["medh5.cache.SUFFIX"].summary


def test_a_dotted_path_resolves_as_python_reads_it() -> None:
    """`medh5.dataset.check` is the function the package exports, not the
    submodule its name shadows."""
    import medh5.dataset

    assert api_index.resolve("medh5.dataset.check") is medh5.dataset.check
    assert api_index.resolve("medh5.dataset.manifest.load") is not None
    assert api_index.resolve("medh5.no_such_module.x") is api_index._MISSING
