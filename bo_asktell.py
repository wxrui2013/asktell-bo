#!/usr/bin/env python3
"""兼容入口：python bo_asktell.py …  等同于 python -m asktell …"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from asktell.core import main

if __name__ == "__main__":
    main()
