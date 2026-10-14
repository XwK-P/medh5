"""The deprecation mechanism (CONTRIBUTING.md, "Deprecating something").

There was no way to deprecate anything before 2.1.0: no warning, no prefix to
filter on, no rule for a module attribute.  These hold the mechanism, and scan
the sources so that every deprecation uses it.
"""

from __future__ import annotations

import re
import types
import warnings

import pytest

from medh5 import _deprecation
from tests.helpers import ROOT

PACKAGE = ROOT / "medh5"
CRATES = ROOT / "crates"


def test_warn_names_3_0_first() -> None:
    with pytest.raises(ValueError, match="starts with"):
        _deprecation.warn("old_name is deprecated")
    with pytest.warns(DeprecationWarning, match="^medh5 3.0 removes x: use y$"):
        _deprecation.warn("medh5 3.0 removes x: use y")
    with pytest.warns(FutureWarning, match="^medh5 3.0 changes"):
        _deprecation.warn("medh5 3.0 changes the default", category=FutureWarning)
    with pytest.raises(TypeError, match="DeprecationWarning or a FutureWarning"):
        _deprecation.warn("medh5 3.0 x", category=UserWarning)


def test_a_warning_points_at_the_callers_code() -> None:
    def deprecated() -> None:
        _deprecation.warn("medh5 3.0 removes deprecated: use fresh")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated()
    assert caught[0].filename == __file__


def test_the_suite_turns_an_unexpected_deprecation_into_an_error() -> None:
    """`filterwarnings` in pyproject.toml: medh5 never uses what it deprecates."""
    with pytest.raises(DeprecationWarning, match="^medh5 3.0 "):
        _deprecation.warn("medh5 3.0 removes x: use y")


def test_a_deprecated_module_attribute_warns_resolves_and_is_not_exported() -> None:
    module = types.ModuleType("medh5_fake")
    module.__all__ = ["fresh"]
    module.fresh = fresh = object()
    module._DEPRECATED = {"old": (fresh, "medh5 3.0 removes old: use fresh")}
    module.__getattr__ = _deprecation.module_getattr(  # type: ignore[method-assign]
        "medh5_fake", module._DEPRECATED
    )
    with pytest.warns(DeprecationWarning, match="^medh5 3.0 removes old: use fresh"):
        assert module.__getattr__("old") is fresh
    assert "old" not in module.__all__
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        module.__getattr__("missing")
    with pytest.raises(ValueError, match="starts with"):
        _deprecation.module_getattr("medh5_fake", {"x": (1, "x is deprecated")})


def test_rust_warns_with_the_same_prefix() -> None:
    source = (CRATES / "medh5-python/src/deprecation.rs").read_text(encoding="utf-8")
    prefix = re.search(r'pub\(crate\) const PREFIX: &str = "([^"]*)";', source)
    assert prefix and prefix[1] == _deprecation.PREFIX


def test_every_rust_deprecation_names_its_release_and_3_0() -> None:
    """`#[deprecated(since = "2.1.0", note = "<what to use>; removed in 3.0")]`."""
    bad = []
    for path in sorted(CRATES.rglob("*.rs")):
        if "vendor" in path.parts or "target" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"#\[deprecated(.*?)\]\s*\n", text, re.S):
            args = match[1]
            since = re.search(r'since\s*=\s*"(\d+)\.(\d+)\.(\d+)"', args)
            note = re.search(r'note\s*=\s*"([^"]*)"', args)
            if (
                since is None
                or tuple(map(int, since.groups())) < (2, 1, 0)
                or note is None
                or "3.0" not in note[1]
            ):
                line = text.count("\n", 0, match.start()) + 1
                bad.append(f"{path.relative_to(ROOT)}:{line}")
    assert not bad, "name since = 2.1.0 or later, and 3.0 in the note: " + str(bad)


def test_every_python_deprecation_message_starts_with_the_prefix() -> None:
    """Every `warn(` call in medh5 and every `_DEPRECATED` table entry."""
    bad = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "_deprecation.py":
            continue
        text = path.read_text(encoding="utf-8")
        where = path.relative_to(ROOT)
        # `_deprecation.warn(` anywhere, and a bare `warn(` where it is imported.
        bare = re.search(r"from \S*_deprecation import [^\n]*\bwarn\b", text)
        call = (
            r"(?:\b_deprecation\.|(?<![.\w]))warn" if bare else r"\b_deprecation\.warn"
        )
        for found in re.finditer(call + r"\(\s*(f?)([\"'])(.*?)\2", text, re.S):
            if not found[3].startswith(_deprecation.PREFIX):
                bad.append(f"{where}: {found[3][:40]!r}")
        table = re.search(r"^_DEPRECATED\b.*?^\}", text, re.S | re.M)
        if table:
            for message in re.findall(r",\s*([\"'])(.*?)\1\s*\)", table[0], re.S):
                if not message[1].startswith(_deprecation.PREFIX):
                    bad.append(f"{where}: {message[1][:40]!r}")
        for raw in re.finditer(
            r"warnings\.warn\([^)]*(DeprecationWarning|FutureWarning)", text, re.S
        ):
            line = text.count("\n", 0, raw.start()) + 1
            bad.append(f"{where}:{line}: use medh5._deprecation.warn")
    assert not bad, bad
