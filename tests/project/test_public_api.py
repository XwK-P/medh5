"""The public API is compared with the last release before it ships.

2.0.2's "no public API change" was checked by hand, and the patch still changed
Rust structs.  CI now compares both APIs with the newest release at or below
this tree: `cargo semver-checks` the crates, `.github/scripts/public_api.py`
the Python package.  These hold the script and its snapshot,
`tests/project/python_api.txt`, which says what a pull request changed in the
Python API where a reviewer reads it.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from tests.helpers import ROOT

SCRIPTS = ROOT / ".github" / "scripts"


def _script(name: str) -> ModuleType:
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `dataclasses` looks the module up while it builds the script's classes.
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def api() -> ModuleType:
    return _script("public_api.py")


def test_the_installed_api_is_the_snapshot(api: ModuleType) -> None:
    """A change to the Python API changes the snapshot in the same pull
    request, where a reviewer sees it."""
    snapshot = api.SNAPSHOT.read_text(encoding="utf-8")
    assert api.dump() == snapshot, (
        "the public Python API differs from tests/project/python_api.txt; if "
        "the change is meant, run `python .github/scripts/public_api.py write`"
    )


# --- dump ---


class Mode(enum.Enum):
    FAST = "fast"


@dataclasses.dataclass
class Settings:
    name: str
    tags: list[str] = dataclasses.field(default_factory=list)


class Thing:
    def method(self, a, /, b, *, c=Mode.FAST, d=object()): ...  # noqa: ANN001,ANN201,B008

    @classmethod
    def build(cls, size=(1, 2), **options): ...  # noqa: ANN001,ANN003,ANN206


def test_signatures_carry_names_kinds_and_literal_defaults(api: ModuleType) -> None:
    """No annotation and no repr that differs between Pythons: an enum member
    prints `Class.NAME`, a dataclass factory `<factory>`, any other value that
    is not a literal `<value>`, and the receiver is dropped."""
    assert api._signature(Thing.method, bound=True) == (
        "(a, /, b, *, c=Mode.FAST, d=<value>)"
    )
    assert api._signature(Thing.build, bound=True) == "(size=(1, 2), **options)"
    assert api._signature(Settings) == "(name, tags=<factory>)"
    assert api._signature(lambda *args, key: None) == "(*args, key)"
    assert api._default(...) == "<value>"  # PyO3's spelling of an unknown default
    assert api._default(float("nan")) == "nan"


def test_a_dump_line_carries_no_address(api: ModuleType) -> None:
    entry = ("medh5.X", "attribute", "<object object at 0x7f3a9c0b2d50>")
    assert api.line(entry) == "medh5.X\tattribute\t<object object>"
    assert api.line(("medh5", "module", "")) == "medh5\tmodule"


PACKAGE = {
    "__init__.py": '''
        import dataclasses
        import enum
        from typing import Optional

        from fakeapi import sub

        __all__ = ["LIMIT", "Box", "Error", "Kind", "Maybe", "measure", "sub"]

        LIMIT = 3
        Maybe = Optional[int]

        class Error(Exception):
            pass

        class Kind(enum.Enum):
            A = 1

        class _Base:
            def shared(self, x):
                pass

            def __len__(self):
                return 0

            def __repr__(self):
                return "Box()"

        @dataclasses.dataclass
        class Box(_Base):
            size: int
            label: str = "x"

            @property
            def area(self):
                return self.size

            class Inner:
                def go(self):
                    pass

        def measure(box, *, unit="mm"):
            pass

        def _old():
            pass

        _DEPRECATED = {"old": (_old, "medh5 3.0 removes old: use measure")}
        ''',
    "sub.py": '__all__ = ["ONE"]\nONE = 1.0\n',
    "_private.py": '__all__ = ["HIDDEN"]\nHIDDEN = 1\n',
}


def test_the_dump_lists_names_members_and_kinds(
    api: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every public module and name; members of our classes and of their
    bases of ours, with the protocol dunders but not the generated ones; a
    deprecated name from the table; no private module or base."""
    package = tmp_path / "fakeapi"
    package.mkdir()
    for name, source in PACKAGE.items():
        (package / name).write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(api, "PACKAGE", "fakeapi")
    assert api.dump().splitlines() == [
        "fakeapi\tmodule",
        "fakeapi.Box\tclass\t(size, label='x') bases=",
        "fakeapi.Box.Inner\tclass\t(?) bases=",
        "fakeapi.Box.Inner.go\tmethod\t()",
        "fakeapi.Box.__len__\tmethod\t()",
        "fakeapi.Box.area\tproperty",
        "fakeapi.Box.label\tfield",
        "fakeapi.Box.shared\tmethod\t(x)",
        "fakeapi.Box.size\tfield",
        "fakeapi.Error\tclass\t(?) bases=Exception",
        "fakeapi.Kind\tclass\t(?) bases=",
        "fakeapi.Kind.A\tmember\t1",
        "fakeapi.LIMIT\tconstant\tint",
        "fakeapi.Maybe\talias",
        "fakeapi.measure\tfunction\t(box, *, unit='mm')",
        "fakeapi.old\tdeprecated",
        "fakeapi.sub\tmodule",
        "fakeapi.sub.ONE\tconstant\tfloat",
    ]


# --- compare ---

BEFORE = """\
medh5\tmodule
medh5.Error\tclass\t(?) bases=Exception
medh5.Writer\tclass\t(path, *, codec='balanced') bases=
medh5.Writer.add\tmethod\t(ann_id, boxes, *, scores=None)
medh5.Writer.close\tmethod\t()
medh5.load\tfunction\t(path, mode='r', **options)
medh5.LIMIT\tconstant\tint
medh5.old\tfunction\t(x)
"""


def _changes(api: ModuleType, edit: dict[str, str | None]) -> tuple[list, list]:
    """BEFORE against BEFORE with lines replaced (None removes one)."""
    after = []
    for text in BEFORE.splitlines():
        path = text.split("\t", 1)[0]
        if path not in edit:
            after.append(text)
    after += [value for value in edit.values() if value is not None]
    return api.changes(api.parse(BEFORE), api.parse("\n".join(after)))


def test_the_same_api_changes_nothing(api: ModuleType) -> None:
    assert api.changes(api.parse(BEFORE), api.parse(BEFORE)) == ([], [])


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (
            {"medh5.Writer.close": None},
            "medh5.Writer.close: the method was removed",
        ),
        (
            {"medh5.load": "medh5.load\tfunction\t(mode='r', path=None, **options)"},
            "medh5.load: positional parameter 1 was 'path', is 'mode'",
        ),
        (
            {"medh5.load": "medh5.load\tfunction\t(path, mode, **options)"},
            "medh5.load: parameter 'mode' lost its default",
        ),
        (
            {"medh5.Writer.add": "medh5.Writer.add\tmethod\t(ann_id, boxes)"},
            "medh5.Writer.add: keyword 'scores' was removed",
        ),
        (
            {"medh5.Writer.add": "medh5.Writer.add\tmethod\t(ann_id, boxes, *, scores)"},
            "medh5.Writer.add: parameter 'scores' lost its default",
        ),
        (
            {"medh5.Writer": "medh5.Writer\tclass\t(path, *, codec='balanced', root) bases="},
            "medh5.Writer: new required keyword 'root'",
        ),
        (
            {"medh5.load": "medh5.load\tfunction\t(path, mode='r', *, strict=False)"},
            "medh5.load: **kwargs was removed",
        ),
        (
            {"medh5.load": "medh5.load\tfunction\t(path, /, mode='r', **options)"},
            "medh5.load: parameter 'path' became positional-only",
        ),
        (
            {"medh5.Error": "medh5.Error\tclass\t(?) bases=ValueError"},
            "medh5.Error: no longer a subclass of Exception",
        ),
        (
            {"medh5.LIMIT": "medh5.LIMIT\tconstant\tstr"},
            "medh5.LIMIT: was int, is str",
        ),
        (
            {"medh5.load": "medh5.load\tclass\t(path, mode='r', **options) bases="},
            "medh5.load: was a function, is a class",
        ),
    ],
)
def test_a_break_is_named(
    api: ModuleType, edit: dict[str, str | None], expected: str
) -> None:
    breaks, _ = _changes(api, edit)
    assert expected in breaks


def test_a_removed_class_is_one_break_not_one_per_member(api: ModuleType) -> None:
    breaks, _ = _changes(
        api,
        {"medh5.Writer": None, "medh5.Writer.add": None, "medh5.Writer.close": None},
    )
    assert breaks == ["medh5.Writer: the class was removed"]


def test_what_a_minor_may_add(api: ModuleType) -> None:
    """A new name, a new keyword with a default, a new optional positional
    after the others, a new base, and a name moving to `deprecated`."""
    breaks, added = _changes(
        api,
        {
            "medh5.Writer.add": (
                "medh5.Writer.add\tmethod\t(ann_id, boxes, extra=None, *, "
                "scores=None, quality=None)"
            ),
            "medh5.Error": "medh5.Error\tclass\t(?) bases=Exception,medh5.Base",
            "medh5.old": "medh5.old\tdeprecated",
            "medh5.Writer.flush": "medh5.Writer.flush\tmethod\t()",
            "medh5.new": "medh5.new\tfunction\t(x, *, y=1)",
            "medh5.Reader": "medh5.Reader\tclass\t(path) bases=",
            "medh5.Reader.read": "medh5.Reader.read\tmethod\t()",
        },
    )
    assert breaks == []
    # A new class's members come with it.
    assert added == [
        "medh5.Reader: new class",
        "medh5.Writer.flush: new method",
        "medh5.new: new function",
    ]


def test_an_unreadable_signature_is_not_compared(api: ModuleType) -> None:
    breaks, _ = _changes(
        api, {"medh5.Writer": "medh5.Writer\tclass\t(?) bases="}
    )
    assert breaks == []


def test_a_deprecated_name_leaves_only_in_a_major(api: ModuleType) -> None:
    deprecated = BEFORE.replace("medh5.old\tfunction\t(x)", "medh5.old\tdeprecated")
    removed = deprecated.replace("medh5.old\tdeprecated\n", "")
    status, report = api.compare(deprecated, removed, "minor")
    assert status == 1
    assert "`medh5.old: the deprecated was removed`" in report
    status, report = api.compare(deprecated, removed, "major")
    assert status == 0
    assert "1 breaking change(s), allowed in a major" in report


def test_a_patch_lists_its_additions(
    api: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A patch adds no API without saying so: the additions go to the job's
    summary, where the release's author reads them."""
    old, new, summary = tmp_path / "old.txt", tmp_path / "new.txt", tmp_path / "s.md"
    old.write_text(BEFORE, encoding="utf-8")
    new.write_text(BEFORE + "medh5.extra\tfunction\t()\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert api.main(["compare", str(old), str(new), "--release-type", "patch"]) == 0
    text = summary.read_text(encoding="utf-8")
    assert "No breaking change." in text
    assert "- `medh5.extra: new function`" in text
    assert api.main(["compare", str(new), str(old), "--release-type", "patch"]) == 1


# --- the baseline ---


@pytest.mark.parametrize(
    ("current", "branch", "override", "expected"),
    [
        # The tree's version is the newest release's: the next release's work.
        ("2.0.3", "main", "", ("2.0.3", "minor")),
        ("2.0.3", "release/2.0", "", ("2.0.3", "patch")),
        # A merge group names its base as a ref.
        ("2.0.3", "refs/heads/release/2.0", "", ("2.0.3", "patch")),
        # A bumped tree is compared with the release before it.
        ("2.1.0", "main", "", ("2.0.3", "minor")),
        ("2.0.4", "release/2.0", "", ("2.0.3", "patch")),
        ("3.0.0", "main", "", ("2.0.3", "major")),
        # A release candidate is below its final, and never a baseline.
        ("2.1.0-rc.2", "main", "", ("2.0.3", "minor")),
        # 3.0's work says so outright.
        ("2.0.3", "main", "major", ("2.0.3", "major")),
    ],
)
def test_the_baseline_is_the_newest_final_release_at_or_below(
    current: str, branch: str, override: str, expected: tuple[str, str]
) -> None:
    release = _script("release.py")
    published = ["1.4.4", "2.0.0", "2.0.1", "2.0.3", "2.1.0-rc.1", "2.0.2", "x"]
    assert (
        release.api_baseline(current, published, branch=branch, override=override)
        == expected
    )


def test_the_baseline_refuses_what_it_cannot_compare() -> None:
    release = _script("release.py")
    with pytest.raises(SystemExit, match="no final release at or below 1.0.0"):
        release.api_baseline("1.0.0", ["2.0.0", "1.0.0-rc.1"], branch="main")
    with pytest.raises(SystemExit, match="is not one of major, minor, patch"):
        release.api_baseline("2.0.3", ["2.0.3"], branch="main", override="huge")
    with pytest.raises(SystemExit, match="is not X.Y.Z"):
        release.api_baseline("2.0", ["2.0.3"], branch="main")
