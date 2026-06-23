"""Compatibility helpers for TPOT 0.12.x and current TPOT APIs.

The TPOT API changed substantially after the 0.12.x series.  This module
keeps project scripts small by detecting the installed TPOT constructor
signature and translating common Spec4ML options to the correct argument names.
"""

from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
