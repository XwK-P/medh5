"""What each layer may do.  Only the engine replaces a file (D23, 1.0 §14.4).

At 2.0.2 one kind of write followed three rules: the engine followed a
symbolic link and kept it, Python's ``write_text`` replaced the link, and the
native command line's ``std::fs::write`` wrote through it in place.  Every file
medh5 writes now goes through the engine's replace routine; these scan the
sources so that no new write site goes around it.  The native command line
has its own guard, `crates/medh5-cli/tests/layers.rs`.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator
from pathlib import Path

import medh5

# The package under test, wherever it is installed: CI tests the built wheel,
# with no `medh5/` beside the tests.
PACKAGE = Path(medh5.__file__).parent

# The write sites other workstreams still own, by file and function: each
# moves to the engine in its owner's pull request and leaves this list, which
# is empty at release.
ALLOWED = {
    ("medh5/task.py", "TaskManifest.save"),  # W21-11 PR 3
    ("medh5/_files.py", "temporary_beside"),  # the later of W21-11 PR 3, W21-13 PR 9
    ("medh5/io/nifti.py", "write_nifti"),  # W21-13 PR 9
    ("medh5/io/nnunetv2.py", "to_nnunetv2"),  # W21-13 PR 9
    ("medh5/io/dicom_seg.py", "to_dicom_seg"),  # W21-13 PR 9
    ("medh5/io/rtstruct.py", "to_rtstruct"),  # W21-13 PR 9
    ("medh5/torch/clinical.py", "ConceptVocabulary.save"),  # W21-14 PR 4
}

_REPLACING = {("os", "replace"), ("os", "rename"), ("os", "open"), ("shutil", "move")}
_SAVING = {"save", "save_as", "to_filename"}


def _name(node: ast.expr) -> str:
    """`a.b.c` for an attribute chain, `""` for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _writes_by_mode(call: ast.Call) -> bool:
    """`open(..., "w" | "x" | "a")`, positionally or as `mode=`."""
    mode: ast.expr | None = call.args[1] if len(call.args) > 1 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    return (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and any(c in mode.value for c in "wxa")
    )


def _is_staged(item: ast.withitem) -> bool:
    call = item.context_expr
    return isinstance(call, ast.Call) and _name(call.func).endswith("stage_file")


class _Writes(ast.NodeVisitor):
    """Each call that writes a file around the engine, with the function it
    is in."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.staged = 0
        self.found: list[tuple[str, int, str]] = []

    def _enter(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _enter

    def visit_With(self, node: ast.With) -> None:
        staged = any(_is_staged(item) for item in node.items)
        self.staged += staged
        self.generic_visit(node)
        self.staged -= staged

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "tempfile":
                self._found(node, "import tempfile")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "tempfile":
            self._found(node, "from tempfile import ...")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _name(node.func)
        head, _, last = name.rpartition(".")
        # A method is found on any receiver, `(root / "x").write_text` too.
        method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if (head, last) in _REPLACING or head.split(".")[0] == "tempfile":
            self._found(node, name)
        elif method in ("write_text", "write_bytes"):
            self._found(node, f"Path.{method}")
        elif name == "open" and _writes_by_mode(node):
            self._found(node, 'open(..., "w")')
        elif method in _SAVING and not self.staged:
            shown = name or f"<expression>.{method}"
            self._found(node, f"{shown} outside `with _core.stage_file(...)`")
        self.generic_visit(node)

    def _found(self, node: ast.AST, what: str) -> None:
        self.found.append((".".join(self.scope), getattr(node, "lineno", 0), what))


def _writes_in(source: str) -> list[tuple[str, int, str]]:
    walker = _Writes()
    walker.visit(ast.parse(source))
    return walker.found


def _sources() -> Iterator[tuple[str, str]]:
    for path in sorted(PACKAGE.rglob("*.py")):
        yield (
            path.relative_to(PACKAGE.parent).as_posix(),
            path.read_text(encoding="utf-8"),
        )


def test_only_the_engine_replaces_files() -> None:
    found = [
        (file, scope, line, what)
        for file, source in _sources()
        for scope, line, what in _writes_in(source)
    ]
    assert any(file == "medh5/_files.py" for file, _ in _sources()), PACKAGE
    outside = [f for f in found if (f[0], f[1]) not in ALLOWED]
    assert outside == [], (
        "write files through the engine (medh5._core.write_atomic, "
        f"medh5._core.stage_file), not around it: {outside}"
    )
    # An entry its owner has moved leaves the list.
    used = {(f[0], f[1]) for f in found}
    assert used >= ALLOWED, f"moved; remove from ALLOWED: {ALLOWED - used}"


def test_the_guard_refuses_what_2_0_2_did() -> None:
    """`_files.write_text` as 2.0.2 wrote it, and the other ways around the
    engine, are each found; a foreign writer inside a staged file is not."""
    old = textwrap.dedent(
        """
        def write_text(target, text):
            temporary, fd = temporary_beside(target)
            with open(fd, "w", encoding="utf-8") as out:
                out.write(text)
            os.replace(temporary, target)
        """
    )
    assert [what for _, _, what in _writes_in(old)] == [
        'open(..., "w")',
        "os.replace",
    ]
    others = textwrap.dedent(
        """
        import tempfile
        def f(path, image):
            path.write_bytes(b"")
            (path / "dataset.json").write_text("{}")
            shutil.move(path, path)
            open(path, mode="a")
            nib.save(image, path)
            with _core.stage_file(path) as staged:
                nib.save(image, staged.path)
            open(path)
            open(path, "rb")
        """
    )
    assert [what for _, _, what in _writes_in(others)] == [
        "import tempfile",
        "Path.write_bytes",
        "Path.write_text",
        "shutil.move",
        'open(..., "w")',
        "nib.save outside `with _core.stage_file(...)`",
    ]
