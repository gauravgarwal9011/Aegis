"""Ensure the project root is importable when pytest is invoked as bare
`pytest` (which, unlike `python -m pytest`, does not add the CWD to sys.path)."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
