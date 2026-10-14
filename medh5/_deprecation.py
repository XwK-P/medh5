"""How medh5 deprecates something that 3.0 removes or changes.

Every message starts with `PREFIX` and says what to use instead, so a user can
filter all of them (``-W "error:medh5 3.0 "``) and the test suite turns every
one into an error unless a test expects it (`filterwarnings` in
`pyproject.toml`). `DeprecationWarning` marks a name or parameter 3.0 removes
or renames; `FutureWarning`, which Python shows to end users too, a default
whose *result* 3.0 changes, raised only where the answer will differ.

A deprecated module attribute leaves `__all__` and moves into the module's
`_DEPRECATED` table, served by `module_getattr`::

    _DEPRECATED = {"old_name": (new_name, "medh5 3.0 removes old_name: use new_name")}
    __getattr__ = module_getattr(__name__, _DEPRECATED)

Reaching the name warns, so a deprecated function needs no second warning in
its body. Warnings raised by classes `_core` binds directly come from
`crates/medh5-python/src/deprecation.rs`, with the same prefix.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["PREFIX", "module_getattr", "warn"]

PREFIX = "medh5 3.0 "


def warn(
    message: str,
    *,
    category: type[Warning] = DeprecationWarning,
    stacklevel: int = 3,
) -> None:
    """Warn about something 3.0 removes or changes; `message` starts with PREFIX.

    The default `stacklevel` points at the caller of the function that calls
    this, which is the user's code for a deprecated function or parameter, and
    for a module attribute served by `module_getattr`.
    """
    if not message.startswith(PREFIX):
        raise ValueError(f"a deprecation message starts with {PREFIX!r}: {message!r}")
    if not issubclass(category, (DeprecationWarning, FutureWarning)):
        raise TypeError(
            f"a deprecation is a DeprecationWarning or a FutureWarning, not {category}"
        )
    warnings.warn(message, category, stacklevel=stacklevel)


def module_getattr(
    module: str, deprecated: Mapping[str, tuple[object, str]]
) -> Callable[[str], Any]:
    """A module `__getattr__` serving `deprecated` names, each with its message."""
    for name, (_, message) in deprecated.items():
        if not message.startswith(PREFIX):
            raise ValueError(f"{module}.{name}: the message starts with {PREFIX!r}")

    def __getattr__(name: str) -> Any:
        try:
            value, message = deprecated[name]
        except KeyError:
            raise AttributeError(
                f"module {module!r} has no attribute {name!r}"
            ) from None
        warn(message)
        return value

    return __getattr__
