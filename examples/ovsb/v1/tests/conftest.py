"""Pytest setup: make ``ovsb`` (recipe dir) and ``wespeaker`` importable."""

import os
import sys

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(RECIPE_DIR, '..', '..', '..'))

for _path in (RECIPE_DIR, REPO_ROOT):  # repo root ends up first
    if _path not in sys.path:
        sys.path.insert(0, _path)
