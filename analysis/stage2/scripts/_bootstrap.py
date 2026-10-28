"""Lets the scripts run as files too (python analysis/stage2/scripts/<name>.py): puts analysis/ on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[2]
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
