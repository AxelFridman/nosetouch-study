"""Helpers shared by the motionlab tests (a uniquely named module: other test folders have conftests too)."""
from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
REPO = ANALYSIS_DIR.parent
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))


def condition(hand: str = 'right', orientation: str = 'upright_toward', speed: str = 'natural',
              variant: str = 'standard', start_position: str = 'natural') -> dict:
    """A TrialCondition."""
    return {'hand': hand, 'orientation': orientation, 'speed': speed, 'start_position': start_position,
            'variant': variant, 'posture': 'standing', 'rep': 0}
