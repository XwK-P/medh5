"""The encoded form of one annotation: datasets plus kind-specific attributes."""

from __future__ import annotations

from medh5 import _core

AnnotationPayload = _core.AnnotationPayload
"""Datasets and kind-specific attributes for one encoded annotation.

What every encoder returns and the writer stores; ``data`` is the ``data``
dataset, ``class_ids`` the encoding order (§6.2), ``stacked_axes`` the leading
axes of ``data`` that get chunk extent 1 (§14.1).
"""

__all__ = ["AnnotationPayload"]
