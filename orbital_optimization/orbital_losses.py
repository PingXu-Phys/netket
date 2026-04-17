"""Canonical loss entry for orbital optimisation.

This module is the new public name for the loss/diagnostics layer.
During the transition away from optimize_no_*, it re-exports the
implementation from optimize_no_losses.py.
"""

from __future__ import annotations

from .optimize_no_losses import *  # noqa: F401,F403
from .optimize_no_losses import __all__  # noqa: F401
