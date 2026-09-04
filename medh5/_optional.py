"""One place that says how to install an optional dependency.

Every converter and bridge had a ``require_<package>`` of its own, each
spelling the same three lines.  They stay, as names callers already import;
this is what they call.
"""

from __future__ import annotations

import importlib
from typing import Any


def require(module: str, *, extra: str, purpose: str) -> Any:
    """Import *module*, or raise an ``ImportError`` naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"{module} is required for {purpose}. Install it with: "
            f"pip install 'medh5[{extra}]'"
        ) from exc


__all__ = ["require"]
