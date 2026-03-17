"""One-shot script to split plagih/trees.py into a plagih/trees/ package.

Run once:  python scripts/split_trees.py

After running, verify with:
    python -c "from plagih.trees import *; print('OK')"
    pytest plagih/test/ -q
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "plagih" / "trees.py"
PKG = ROOT / "plagih" / "trees"

assert SRC.exists(), f"Source not found: {SRC}"

lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
total = len(lines)
print(f"Read {total} lines from {SRC}")

# --- Boundary definitions (1-based, inclusive) ---
# Section: helpers (imports, RoundDummy, type-guard functions, etc.)
#   lines 1..80
# Section: nodes (Node ABC + all operators + free functions up to Candidate)
#   lines 82..2827  (includes d_sym2node, sympy_expression_check_raise, expr_sympify)
# Section: candidate (Candidate class + helper functions)
#   lines 2829..3008
# Section: node_select
#   lines 3010..3143
# Section: evolution
#   lines 3145..3708
# Section: gp_engine (_ClipAutocast, ExplainableGP, etc.)
#   lines 3710..end


def extract(start_1based, end_1based):
    """Extract lines [start, end] (1-based inclusive)."""
    return lines[start_1based - 1 : end_1based]


# --- Build file contents ---
HELPERS_LINES = extract(1, 80)  # everything up to line 80
NODES_LINES = extract(82, 2827)
CANDIDATE_LINES = extract(2829, 3008)
NODESELECT_LINES = extract(3010, 3143)
EVOLUTION_LINES = extract(3145, 3708)
GP_ENGINE_LINES = extract(3710, total)


# --- Imports for each module ---
HELPERS_HEADER = '''\
"""Helper functions and sympy extensions for plagih trees."""

import copy
import random
import time
import warnings
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeGuard, Union, cast

import pandas as pd
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.utilities.exceptions import ignore_warnings

from plagih.config import cfg as _cfg
from plagih.util import *

'''

# For nodes.py: needs helpers + stdlib
NODES_HEADER = '''\
"""Node hierarchy for plagih GP trees.

Contains the Node ABC, all operator and terminal classes, and related
free functions (sympy_to_tree, tree_simplification, etc.).
"""

import copy
import random
import time
import warnings
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeGuard, Union, cast

import pandas as pd
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.utilities.exceptions import ignore_warnings

from plagih.config import cfg as _cfg
from plagih.util import *
from plagih.trees._helpers import RoundDummy, is_terminal, is_number

'''

CANDIDATE_HEADER = '''\
"""Candidate class and selection helpers for plagih GP."""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import sympy

from plagih.config import cfg as _cfg
from plagih.util import *
from plagih.trees._helpers import is_terminal, is_number
from plagih.trees.nodes import (
    BaseOperator, Boolean, ChainableOp, Node, NodeWithChilds, Number, Symbol,
    d_sym2node, d_sym2node_chain,
)

'''

NODESELECT_HEADER = '''\
"""NodeSelect — random node selection for tree construction."""

import random
from typing import Any, Dict, List, Optional, Tuple, Type

import sympy

from plagih.config import cfg as _cfg
from plagih.util import *
from plagih.trees.nodes import Boolean, Node, Number, Symbol
from plagih.trees.candidate import norm_choices, operatorpool_to_picks

'''

EVOLUTION_HEADER = '''\
"""Evolution — mutation, crossover, simplification for GP trees."""

import copy
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import sympy

from plagih.config import cfg as _cfg
from plagih.util import *
from plagih.trees._helpers import is_terminal
from plagih.trees.nodes import (
    Abs, Add, And, BaseOperator, Div, Le, Log, Lt, Max, Min, Mul, Node,
    NodeWithChilds, Not, Number, Or, Scale, Sign, Sin, Sqrt, Square, Symbol,
    Terminal, node_deepcopy, tree_simplification, cast_input, eval_parsimony,
)
from plagih.trees.candidate import Candidate, check_operator_pool, selection_tournament
from plagih.trees.node_select import NodeSelect

'''

GP_ENGINE_HEADER = '''\
"""ExplainableGP — the main GP engine class."""

import copy
import pickle
import random
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import sympy

from plagih.config import cfg as _cfg
from plagih.evaluation_context import EvaluationContext
from plagih.monitoring import GPMonitor
from plagih.parallel import ParallelEvaluator, Strategy
from plagih.paretofront import *
from plagih.tree_complexity.tree_edit_distance import *
from plagih.tree_complexity.python_bytecode_complexity import *
from plagih.util import *
from plagih.trees._helpers import is_terminal, is_number
from plagih.trees.nodes import (
    Add, Node, Number, Symbol, Terminal, node_deepcopy, tree_simplification,
    cast_input, eval_parsimony, expr_sympify,
)
from plagih.trees.candidate import Candidate, selection_tournament
from plagih.trees.node_select import NodeSelect
from plagih.trees.evolution import Evolution, population_statistics

'''


# --- Strip original imports from section bodies ---
# The node section starts at line 82 which is `class RoundDummy`, no imports to strip
# Candidate, NodeSelect, Evolution, GP_Engine sections also start at class defs


def strip_leading_imports(body_lines):
    """Remove any leading import/from lines and blank lines at the top."""
    result = []
    past_imports = False
    for line in body_lines:
        if not past_imports:
            s = line.strip()
            if s.startswith("import ") or s.startswith("from ") or s == "" or s.startswith("#"):
                continue
            past_imports = True
        result.append(line)
    return result


# --- Actually write the files ---
# 1. Backup original
backup = SRC.with_suffix(".py.bak")
if not backup.exists():
    shutil.copy2(SRC, backup)
    print(f"Backed up original to {backup}")

# 2. Create package directory
PKG.mkdir(exist_ok=True)
print(f"Created package directory: {PKG}")

# 3. Write _helpers.py
# We keep the original module docstring + imports + helper functions
(PKG / "_helpers.py").write_text(
    HELPERS_HEADER + "".join(extract(56, 80)),  # np.set_printoptions + helpers
    encoding="utf-8",
)
print("  _helpers.py written")

# 4. Write nodes.py
(PKG / "nodes.py").write_text(
    NODES_HEADER + "".join(NODES_LINES),
    encoding="utf-8",
)
print("  nodes.py written")

# 5. Write candidate.py
(PKG / "candidate.py").write_text(
    CANDIDATE_HEADER + "\n" + "".join(CANDIDATE_LINES),
    encoding="utf-8",
)
print("  candidate.py written")

# 6. Write node_select.py
(PKG / "node_select.py").write_text(
    NODESELECT_HEADER + "\n" + "".join(NODESELECT_LINES),
    encoding="utf-8",
)
print("  node_select.py written")

# 7. Write evolution.py
(PKG / "evolution.py").write_text(
    EVOLUTION_HEADER + "\n" + "".join(EVOLUTION_LINES),
    encoding="utf-8",
)
print("  evolution.py written")

# 8. Write gp_engine.py
(PKG / "gp_engine.py").write_text(
    GP_ENGINE_HEADER + "\n" + "".join(GP_ENGINE_LINES),
    encoding="utf-8",
)
print("  gp_engine.py written")

# 9. Write __init__.py
init_content = '''\
"""plagih.trees — GP expression-tree package.

This package was split from a monolithic ``trees.py`` into sub-modules.
All public names are re-exported here so that ``from plagih.trees import *``
and ``from plagih.trees import Node, Add, ExplainableGP`` continue to work.
"""

from plagih.trees._helpers import *  # noqa: F401,F403
from plagih.trees.nodes import *  # noqa: F401,F403
from plagih.trees.candidate import *  # noqa: F401,F403
from plagih.trees.node_select import *  # noqa: F401,F403
from plagih.trees.evolution import *  # noqa: F401,F403
from plagih.trees.gp_engine import *  # noqa: F401,F403
'''
(PKG / "__init__.py").write_text(init_content, encoding="utf-8")
print("  __init__.py written")

# 10. Remove old trees.py (it's now a directory)
SRC.unlink()
print(f"Removed old {SRC}")

print("\nDone! Verify with:")
print("  python -c \"from plagih.trees import *; print('OK')\"")
print("  pytest plagih/test/ -q")
