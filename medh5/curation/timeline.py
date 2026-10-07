"""Timepoints: the sample's observation occasions (spec §3.7).

A sample is one subject at one **or more** timepoints.  The declaration lives in
``/meta -> timepoints``; every grid names one, and images, annotations and
transforms inherit theirs rather than repeating it.  Binding time to the *grid*
is what keeps the rule single-valued: a grid belongs to exactly one acquisition
occasion, whereas an image or an annotation might plausibly be argued either
way.

``days_from_baseline`` rather than ``date`` is what models should consume: the
interval survives de-identification date shifting, and it is the clinically
load-bearing quantity.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from medh5._core import TIMEPOINT_FIELDS, Timeline, Timepoint

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")
"""An ISO-8601 calendar date prefix, as ``date`` must start."""

Sequence.register(Timeline)

__all__ = ["DATE_PATTERN", "TIMEPOINT_FIELDS", "Timeline", "Timepoint"]
