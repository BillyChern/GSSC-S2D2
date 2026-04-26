"""Shared pytest configuration."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow tests to import the package without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
