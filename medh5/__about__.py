"""Package identity: the engine's version and the format version it writes.

Both come from the compiled engine, whose version is the Cargo workspace's ---
the one number stamped on the wheel, into every file's ``generator`` and into
every manifest.
"""

from __future__ import annotations

from medh5._core import __format_version__, __version__

__all__ = ["__format_version__", "__version__"]
