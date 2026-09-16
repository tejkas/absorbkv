# SPDX-License-Identifier: Apache-2.0
"""Make ``src/`` importable without an editable install."""

# Standard
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
