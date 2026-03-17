"""One-shot migration: Split plagih/trees.py into plagih/trees/ package.

Run once:  python scripts/migrate_trees_package.py
Then:      pytest plagih/test/ -q
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "plagih"
SRC = ROOT / "trees.py"
BAK = ROOT / "trees.py.bak"
PKG = ROOT / "trees"

# ---------------------------------------------------------------------------
# 1. Read current file
# ---------------------------------------------------------------------------
assert SRC.exists(), f"Source not found: {SRC}"
content = SRC.read_text(encoding="utf-8")
lines = content.splitlines(keepends=True)
total = len(lines)
print(f"Read {total} lines from {SRC}")


# ---------------------------------------------------------------------------
# 2. Find split points by searching for unique marker strings
# ---------------------------------------------------------------------------
def find_line_1based(marker: str) -> int:
    for i, line in enumerate(lines):
        if marker in line:
            return i + 1  # 1-based
    raise ValueError(f"Marker not found: {marker!r}")


split_evo = find_line_1based("class Candidate:")  # start of _evolution.py
split_gp = find_line_1based("# Picklable helper callables for ProcessPoolExecutor")

# Back up to section comment "# ===" before the GP split
for i in range(split_gp - 2, max(split_gp - 6, 0), -1):  # 0-based
    if lines[i].startswith("# ====="):
        split_gp = i + 1  # 1-based
        break

print(f"Split points: _evolution.py starts at line {split_evo}, _gp_engine.py starts at line {split_gp}")

# ---------------------------------------------------------------------------
# 3. Extract sections (1-based inclusive ranges → 0-based slices)
# ---------------------------------------------------------------------------
nodes_lines = lines[: split_evo - 1]  # lines 1 .. split_evo-1
evo_lines = lines[split_evo - 1 : split_gp - 1]  # lines split_evo .. split_gp-1
gp_lines = lines[split_gp - 1 :]  # lines split_gp .. end

print(f"Section sizes: _nodes={len(nodes_lines)}, _evolution={len(evo_lines)}, _gp_engine={len(gp_lines)}")

# ---------------------------------------------------------------------------
# 4. Build file contents
# ---------------------------------------------------------------------------

# --- _nodes.py: keep original imports, strip only evolution/GP-only imports ---
# The original imports work for _nodes.py minus GPMonitor and paretofront
nodes_src = "".join(nodes_lines)
# Remove imports that are only needed by _gp_engine
nodes_src = nodes_src.replace("from plagih.monitoring import GPMonitor\n", "")
nodes_src = nodes_src.replace("from plagih.paretofront import *\n", "")
# Remove the time import (only used by ExplainableGP)
nodes_src = nodes_src.replace("import time\n", "")
# Remove deque import (only used by Candidate)
nodes_src = nodes_src.replace("from collections import deque\n", "")

# --- _evolution.py: add header with imports ---
evo_header = '''\
"""Evolution module: Candidate, NodeSelect, Evolution, and population helpers."""

import copy
import random
import warnings
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import sympy
from sympy.utilities.exceptions import ignore_warnings

from plagih.config import cfg as _cfg
from plagih.util import *
from plagih.trees._nodes import *


'''
evo_src = evo_header + "".join(evo_lines)

# --- _gp_engine.py: add header with imports ---
gp_header = '''\
"""GP Engine module: ExplainableGP and picklable helper callables."""

import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
import sympy

from plagih.config import cfg as _cfg
from plagih.monitoring import GPMonitor
from plagih.paretofront import *
from plagih.util import *
from plagih.trees._nodes import *
from plagih.trees._evolution import *


'''
gp_src = gp_header + "".join(gp_lines)

# --- __init__.py ---
init_src = '''\
"""plagih.trees — Node hierarchy, evolution, and GP engine.

Re-exports all public names so that ``from plagih.trees import *``
and ``from plagih.trees import Node, Add, ExplainableGP`` continue to work.
"""

from plagih.trees._nodes import *       # noqa: F401,F403
from plagih.trees._evolution import *   # noqa: F401,F403
from plagih.trees._gp_engine import *   # noqa: F401,F403
'''

# ---------------------------------------------------------------------------
# 5. Delete old file and create package
# ---------------------------------------------------------------------------
# Ensure backup exists
if not BAK.exists():
    shutil.copy2(SRC, BAK)
    print(f"Created backup: {BAK}")

# Delete old trees.py
SRC.unlink()
print(f"Deleted {SRC}")

# Clean pycache
pycache = ROOT / "__pycache__"
if pycache.exists():
    for f in pycache.glob("trees*"):
        f.unlink()
        print(f"Deleted cached: {f}")

# Create package directory
PKG.mkdir(exist_ok=True)
print(f"Created directory: {PKG}")


# ---------------------------------------------------------------------------
# 6. Write files
# ---------------------------------------------------------------------------
def write_mod(name: str, src: str):
    path = PKG / name
    path.write_text(src, encoding="utf-8")
    n = len(src.splitlines())
    print(f"  Wrote {path.name:20s} ({n:5d} lines)")


write_mod("__init__.py", init_src)
write_mod("_nodes.py", nodes_src)
write_mod("_evolution.py", evo_src)
write_mod("_gp_engine.py", gp_src)

print("\nDone!  Verify with:")
print("  python -c \"from plagih.trees import *; print('OK')\"")
print("  pytest plagih/test/ -q")
