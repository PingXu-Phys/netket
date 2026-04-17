"""Canonical optimizer entry for orbital optimisation.

This module is the new public name for the shared optimization core.
During the transition away from optimize_no_*, it re-exports the
implementation from optimize_no_core.py.
"""

from __future__ import annotations

from .optimize_no_core import *  # noqa: F401,F403
from .optimize_no_core import __all__  # noqa: F401
