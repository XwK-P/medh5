"""Support ``python -m medh5.cli``."""

from __future__ import annotations

import sys

from medh5.cli import main

if __name__ == "__main__":
    sys.exit(main())
