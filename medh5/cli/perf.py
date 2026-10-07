"""The dataloader throughput run, on behalf of the native ``medh5 bench``.

Everything else ``bench`` and ``recompress`` do is the engine's; this one
measures the PyTorch dataloader end to end, so the native command line hands
it back to the package.
"""

from __future__ import annotations

from typing import Any


def throughput(
    path: str, patch: int, workers: int, annotation: str | None
) -> dict[str, Any]:
    """Sustained patches/s through the real dataloader, as a measurement record.

    Raises ``ImportError`` without PyTorch; the command line reports that as a
    skipped measurement rather than a failure.
    """
    from medh5.bench import throughput as measure

    found: dict[str, Any] = measure(
        [path], patch=patch, workers=workers, annotation=annotation
    ).to_json()
    return found


__all__ = ["throughput"]
