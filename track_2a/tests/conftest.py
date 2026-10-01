import json
import sys
from pathlib import Path

import pytest

TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK / "src"))
