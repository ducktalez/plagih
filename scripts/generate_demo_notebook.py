"""
Generates docs/demo.ipynb as a valid nbformat-4 Jupyter notebook.
Run from the project root:  python scripts/generate_demo_notebook.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "docs" / "demo.ipynb"


import uuid


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": _new_id(), "metadata": {}, "source": [src]}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": _new_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [src],
    }


cells = []

# ── Title / intro ──────────────────────────────────────────────────────────────
cells.append(
    md(
        "# plagih — Feature Showcase\n"
        "\n"
        "**plagih** is a symbolic Genetic Programming (GP) framework for Explainable AI.\n"
        "It evolves _expression trees_ — compact, human-readable symbolic formulas — to fit\n"
        "training data, guided by both accuracy and tree complexity (parsimony).\n"
        "\n"
        "This notebook walks through every major feature with self-contained, hand-crafted\n"
        "examples. Run it cell-by-cell to see inline tree visualisations and verify that\n"
        "each subsystem works as expected after code changes.\n"
        "\n"
        "**Sections**\n"
        "1. [Building Blocks](#part1) — terminals, operators, tree inspection\n"
        "2. [Genetic Operations](#part2) — random creation, crossover, mutation\n"
        "3. [Simplification & SymPy Bridge](#part3) — algebraic simplification, roundtrip\n"
        "4. [Evaluation & Complexity](#part4) — NumPy eval, parsimony, TED, bytecode\n"
        "5. [Population & Pareto Front](#part5) — candidates, selection, Pareto filter\n"
        "6. [Monitoring](#part6) — `GPMonitor`, metrics, plots\n"
        "7. [Targeted Optimization](#part7) — `ifte_component_scores`, score-coloured trees\n"
        "\n"
        "> **Note:** Outputs are stripped from version control (`nbstripout`).\n"
        "> Run `python -m jupyter notebook docs/demo.ipynb` to regenerate them locally."
    )
)

# ── Setup ──────────────────────────────────────────────────────────────────────
cells.append(
    code(
        "%matplotlib inline\n"
        "import sys, pathlib\n"
        "# Ensure project root is on the path when running from docs/\n"
        "sys.path.insert(0, str(pathlib.Path().resolve().parent))\n"
        "\n"
        "import copy\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import sympy\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from plagih.demo_helpers import (\n"
        "    make_sample_df, make_cartpole_df,\n"
        "    make_tree_simple, make_tree_trig, make_tree_ifte, make_tree_boolean,\n"
        "    make_tree_simplifiable, make_tree_redundant,\n"
        "    make_tree_crossover_parent_a, make_tree_crossover_parent_b,\n"
        "    make_tree_cartpole, make_tree_ifte_cartpole,\n"
        "    make_evolution, make_cartpole_evolution, do_crossover,\n"
        "    show_tree, show_trees, show_tree_with_scores, make_ifte_node_scores,\n"
        ")\n"
        "\n"
        "print('Setup complete.')"
    )
)

# ── Part 1 ─────────────────────────────────────────────────────────────────────
cells.append(
    md(
        '<a id="part1"></a>\n'
        "## Part 1 — Building Blocks\n"
        "\n"
        "Every plagih expression tree is built from three kinds of **nodes**:\n"
        "\n"
        "| Category | Classes | Output type |\n"
        "|---|---|---|\n"
        "| **Terminals** | `Symbol`, `Number`, `Boolean` | float or bool |\n"
        "| **Math operators** | `Add`, `Mul`, `Sin`, `Sqrt`, `Ifte`, … | float |\n"
        "| **Logic / relational** | `And`, `Or`, `Not`, `Lt`, `Le`, `Eq`, … | bool |\n"
        "\n"
        "Nodes compose recursively: every operator holds its children in `childs`.\n"
        "The `xtype` class attribute encodes the type signature `((input_types,), output_type)`."
    )
)

cells.append(
    code(
        "# --- 1.1 Terminals ---\n"
        "from plagih.trees import Symbol, Number, Boolean\n"
        "\n"
        "a = Symbol(sympy.Symbol('a', real=True))\n"
        "x = Number(sympy.Float(3.14, 6))\n"
        "flag = Boolean(True)\n"
        "\n"
        "print('Symbol  :', a, '  xtype out:', a.get_xtype_self())\n"
        "print('Number  :', x, '  value   :', x.get_value())\n"
        "print('Boolean :', flag, '  xtype out:', flag.get_xtype_self())"
    )
)

cells.append(
    code(
        "# --- 1.2 Math operators — build sin(a*b + 2) by hand ---\n"
        "from plagih.trees import Add, Mul, Sin\n"
        "\n"
        "b = Symbol(sympy.Symbol('b', real=True))\n"
        "tree_trig = make_tree_trig()   # Sin(Add(Mul(a, b), 2))\n"
        "\n"
        "print('Tree (compact)  :', tree_trig)\n"
        "print('SymPy expression:', tree_trig.get_sympy_expr())\n"
        "print('Node count      :', len(tree_trig))\n"
        "show_tree(tree_trig, title='sin(a\u00b7b + 2)')"
    )
)

cells.append(
    code(
        "# --- 1.3 Logic & Ifte operators ---\n"
        "from plagih.trees import And, Ifte, Lt, Not\n"
        "\n"
        "tree_bool = make_tree_boolean()   # And(a < 1, NOT (b > 0))\n"
        "tree_ifte = make_tree_ifte()      # Ifte(a < 0, -1, 1) \u2192 sign function\n"
        "\n"
        "show_trees(\n"
        "    [(tree_bool, 'And(a<1, NOT(b>0))'),\n"
        "     (tree_ifte, 'Ifte(a<0, -1, 1)')],\n"
        "    suptitle='Logic and conditional trees'\n"
        ")"
    )
)

cells.append(
    code(
        "# --- 1.4 Tree inspection ---\n"
        "tree = make_tree_trig()\n"
        "\n"
        "print('str_as_list    :', tree.str_as_list())\n"
        "print('Preorder nodes :', [type(n).__name__ for n in tree.to_preorder()])\n"
        "print('Postorder nodes:', [type(n).__name__ for n in tree.to_postorder()])\n"
        "print('Mutable nodes  :', [type(n).__name__ for n in tree.list_mutable_nodes()])\n"
        "print('LUT id         :', tree.get_lut_id()[:40], '...')"
    )
)

# ── Part 2 ─────────────────────────────────────────────────────────────────────
cells.append(
    md(
        '<a id="part2"></a>\n'
        "## Part 2 — Genetic Operations\n"
        "\n"
        "The `Evolution` class manages all tree-modifying operations:\n"
        "\n"
        "| Operation | Method | What it does |\n"
        "|---|---|---|\n"
        "| Random creation | `evolve_new_tree_depth` | Build a random tree up to a depth limit |\n"
        "| Subtree crossover | `evolve_crossover` | Swap compatible subtrees between two parents |\n"
        "| Point mutation | `evolve_mutate_point` | Replace one node with another of the same type |\n"
        "| Branch mutation | `evolve_mutate_branch_depth` | Replace a random subtree with a new random branch |\n"
        "| Gaussian mutation | `evolve_mutate_filter` | Add Gaussian noise to numeric constants |\n"
        "\n"
        "All methods return deep copies or modify in-place depending on the variant."
    )
)

cells.append(
    code(
        "# --- 2.1 Random tree creation ---\n"
        "import random; random.seed(7); np.random.seed(7)\n"
        "\n"
        "evo = make_evolution()\n"
        "rand_tree = evo.evolve_new_tree_depth(xt_out=float, depth_goal=3)\n"
        "rand_tree.repair_all(depth=0)\n"
        "\n"
        "print('Random tree:', rand_tree.get_sympy_expr())\n"
        "print('Nodes      :', len(rand_tree))\n"
        "show_tree(rand_tree, title=f'Random (depth\u22643)  \u2014  {rand_tree}')"
    )
)

cells.append(
    code(
        "# --- 2.2 Subtree Crossover ---\n"
        "# Parent A: sin(a) + b    Parent B: a * |b|\n"
        "# One random compatible subtree is swapped between the two parents.\n"
        "\n"
        "random.seed(42); np.random.seed(42)\n"
        "\n"
        "parent_a = make_tree_crossover_parent_a()   # sin(a) + b\n"
        "parent_b = make_tree_crossover_parent_b()   # a * |b|\n"
        "child_a, child_b = do_crossover(parent_a, parent_b, evo=evo)\n"
        "\n"
        "show_trees(\n"
        "    [\n"
        "        (parent_a, f'Parent A\\n{parent_a.get_sympy_expr()}'),\n"
        "        (parent_b, f'Parent B\\n{parent_b.get_sympy_expr()}'),\n"
        "        (child_a,  f'Child A\\n{child_a.get_sympy_expr()}'),\n"
        "        (child_b,  f'Child B\\n{child_b.get_sympy_expr()}'),\n"
        "    ],\n"
        "    suptitle='Subtree Crossover \u2014 a compatible branch is swapped'\n"
        ")"
    )
)

cells.append(
    code(
        "# --- 2.3 Mutation variants ---\n"
        "random.seed(0); np.random.seed(0)\n"
        "\n"
        "base = make_tree_trig()   # sin(a*b + 2)\n"
        "\n"
        "# Point mutation: swap one operator/terminal for another of same type\n"
        "point = evo.evolve_mutate_point(copy.deepcopy(base))\n"
        "point.repair_all(depth=0)\n"
        "\n"
        "# Branch mutation: replace a random subtree with a new random branch\n"
        "branch = evo.evolve_mutate_branch_depth(copy.deepcopy(base), depth_goal=2)\n"
        "branch.repair_all(depth=0)\n"
        "\n"
        "# Gaussian filter mutation: add small noise to all numeric constants\n"
        "gauss = copy.deepcopy(base)\n"
        "evo.evolve_mutate_filter(gauss)\n"
        "gauss.repair_all(depth=0)\n"
        "\n"
        "show_trees(\n"
        "    [\n"
        "        (base,   f'Original\\n{base}'),\n"
        "        (point,  f'Point mutation\\n{point}'),\n"
        "        (branch, f'Branch mutation\\n{branch}'),\n"
        "        (gauss,  f'Gaussian mutation\\n{gauss}'),\n"
        "    ],\n"
        "    suptitle='Mutation variants on sin(a\u00b7b + 2)',\n"
        "    figsize_per=(4, 3.5)\n"
        ")"
    )
)

# ── Part 3 ─────────────────────────────────────────────────────────────────────
cells.append(
    md(
        '<a id="part3"></a>\n'
        "## Part 3 — Simplification & SymPy Bridge\n"
        "\n"
        "plagih uses **SymPy** as its symbolic algebra engine.\n"
        "The simplification pipeline has three stages:\n"
        "\n"
        "1. **`tree \u2192 SymPy`** via `get_sympy_expr()` \u2014 lossless symbolic conversion  \n"
        "2. **SymPy algebra** \u2014 `sympy.simplify`, `expand`, `factor`, \u2026  \n"
        "3. **`SymPy \u2192 tree`** via `sympy_to_tree()` \u2014 reconstruct a plagih tree  \n"
        "4. **`tree_node_grouping()`** \u2014 post-process: `Pow(a,2) \u2192 Square(a)`, chain rewriting, \u2026  \n"
        "\n"
        "`tree_simplification()` wraps all four stages with safety guards (size guard,\n"
        "semantic guard) so the pipeline never silently grows or distorts a tree."
    )
)

cells.append(
    code(
        "# --- 3.1 tree_simplification: before / after ---\n"
        "from plagih.trees import tree_simplification\n"
        "\n"
        "bloated = make_tree_simplifiable()   # 1*a + a**2\n"
        "simplified = tree_simplification(copy.deepcopy(bloated), allow_chain=False)\n"
        "\n"
        "print('Before :', bloated.get_sympy_expr(), ' \u2014 nodes:', len(bloated))\n"
        "print('After  :', simplified.get_sympy_expr(), ' \u2014 nodes:', len(simplified))\n"
        "\n"
        "show_trees(\n"
        "    [(bloated, f'Before ({len(bloated)} nodes)\\n1\u00b7a + a\u00b2'),\n"
        "     (simplified, f'After ({len(simplified)} nodes)\\n{simplified.get_sympy_expr()}')],\n"
        "    suptitle='tree_simplification'\n"
        ")"
    )
)

cells.append(
    code(
        "# --- 3.2 tree_node_grouping: structural rewrites ---\n"
        "# Demonstrates Pow(a,2) \u2192 Square, trivial factor removal, etc.\n"
        "from plagih.trees import Pow, Square\n"
        "\n"
        "redundant = make_tree_redundant()   # (a+0)*(b*1) - 0\n"
        "print('Before grouping:', redundant.get_sympy_expr(), '  nodes:', len(redundant))\n"
        "\n"
        "simp = tree_simplification(copy.deepcopy(redundant), allow_chain=False)\n"
        "print('After  grouping:', simp.get_sympy_expr(), '  nodes:', len(simp))\n"
        "\n"
        "show_trees(\n"
        "    [(redundant, f'Before\\n(a+0)\u00b7(b\u00b71)\u22120'),\n"
        "     (simp, f'After\\n{simp.get_sympy_expr()}')],\n"
        "    suptitle='Algebraic reduction via SymPy + tree_node_grouping'\n"
        ")"
    )
)

cells.append(
    code(
        "# --- 3.3 SymPy roundtrip: get_sympy_expr \u2192 sympy_to_tree ---\n"
        "# The roundtrip is the foundation of simplification.\n"
        "# SymPy may rewrite the expression (e.g. reorder terms, factor, cancel),\n"
        "# producing a structurally different but semantically equivalent tree.\n"
        "from plagih.trees import sympy_to_tree\n"
        "\n"
        "tree = make_tree_trig()   # sin(a*b + 2)\n"
        "sympy_expr = tree.get_sympy_expr()\n"
        "print('SymPy expression:', sympy_expr)\n"
        "\n"
        "roundtrip = sympy_to_tree(sympy_expr, allow_chain=False)\n"
        "roundtrip.repair_all(depth=0)\n"
        "print('Roundtrip tree  :', roundtrip)\n"
        "print('Semantically eq :', str(tree.get_sympy_expr()) == str(roundtrip.get_sympy_expr()))\n"
        "\n"
        "show_trees(\n"
        "    [(tree, f'Original\\n{tree}'),\n"
        "     (roundtrip, f'After roundtrip\\n{roundtrip}')],\n"
        "    suptitle='SymPy roundtrip: get_sympy_expr \u2192 sympy_to_tree'\n"
        ")"
    )
)

# ── Part 4 ─────────────────────────────────────────────────────────────────────
cells.append(
    md(
        '<a id="part4"></a>\n'
        "## Part 4 — Evaluation & Complexity\n"
        "\n"
        "plagih trees can be evaluated in three ways:\n"
        "\n"
        "| Method | Speed | Use case |\n"
        "|---|---|---|\n"
        "| `eval_predict_numpy_now(df)` | Fast, eager | Default evaluation on DataFrames |\n"
        "| `get_sympy_expr()` + `lambdify` | Moderate | Exact symbolic evaluation |\n"
        "| `EvaluationContext` | Unified | Multi-mode comparison, LUT caching |\n"
        "\n"
        "**Complexity metrics** measure how much compute a tree represents:\n"
        "\n"
        "| Metric | `complexity_metric` name |\n"
        "|---|---|\n"
        "| Raw node count | `'tree_node_count_raw'` |\n"
        "| Fair node count (weights) | `'tree_node_count_fair'` |\n"
        "| Tree Edit Distance (TED) | `'tree_edit_distance'` |\n"
        "| Python bytecode ops | `'tree_python_bytecode_count'` |\n"
        "| Weighted bytecode ops | `'tree_python_bytecode_weighted_count'` |"
    )
)

cells.append(
    code(
        "# --- 4.1 NumPy evaluation ---\n"
        "df = make_sample_df()   # a=[1..5], b=[-1, 0.5, 2, -0.5, 1]\n"
        "\n"
        "tree = make_tree_trig()   # sin(a*b + 2)\n"
        "results = tree.eval_predict_numpy_now(df)\n"
        "\n"
        "print('Input DataFrame:')\n"
        "print(df.to_string(index=False))\n"
        "print()\n"
        "print('sin(a\u00b7b + 2)  =', np.round(results, 4))"
    )
)

cells.append(
    code(
        "# --- 4.2 EvaluationContext: unified multi-mode evaluation ---\n"
        "# evaluate() with multiple modes returns a dict: {mode: result}\n"
        "from plagih.evaluation_context import EvaluationContext\n"
        "\n"
        "ctx = EvaluationContext(modes=['numpy_eager', 'sympy'], use_lut=True)\n"
        "results = ctx.evaluate(tree, df)   # returns dict when >1 mode\n"
        "\n"
        "print('numpy_eager result:', np.round(results['numpy_eager'], 4))\n"
        "print('sympy expr        :', results['sympy'])"
    )
)

cells.append(
    code(
        "# --- 4.3 Parsimony / complexity metrics ---\n"
        "from plagih.trees import eval_parsimony\n"
        "\n"
        "tree_a = make_tree_simple()   # Add(a, 1)      \u2014 3 nodes\n"
        "tree_b = make_tree_trig()     # sin(a*b + 2)   \u2014 5 nodes\n"
        "\n"
        "for t, label in [(tree_a, 'Add(a,1)'), (tree_b, 'sin(a\u00b7b+2)')]:\n"
        "    for metric in ['tree_node_count_raw', 'tree_node_count_fair',\n"
        "                   'tree_python_bytecode_count']:\n"
        "        p = eval_parsimony(t, complexity_measure=metric)\n"
        "        print(f'  {label:20s}  {metric:35s}  \u2192  {p}')"
    )
)

cells.append(
    code(
        "# --- 4.4 Tree Edit Distance (Zhang-Shasha) ---\n"
        "# TED measures structural similarity between two trees.\n"
        "# 0 = identical; higher = more edits needed to transform one into the other.\n"
        "\n"
        "from plagih.tree_complexity.tree_edit_distance import TedConfig\n"
        "\n"
        "tree_a = make_tree_simple()   # Add(a, 1)\n"
        "tree_b = make_tree_trig()     # sin(a*b + 2)\n"
        "tree_c = copy.deepcopy(tree_a)  # identical copy\n"
        "tree_c.repair_all(depth=0)\n"
        "\n"
        "ted_ab = tree_a.compute_ted(tree_b)\n"
        "ted_aa = tree_a.compute_ted(tree_c)\n"
        "\n"
        "print(f'TED(Add(a,1),  sin(a\u00b7b+2)) = {ted_ab.distance}   (very different)')\n"
        "print(f'TED(Add(a,1),  Add(a,1))   = {ted_aa.distance}   (identical)')"
    )
)

# ── Part 5 ─────────────────────────────────────────────────────────────────────
cells.append(
    md(
        '<a id="part5"></a>\n'
        "## Part 5 — Population & Pareto Front\n"
        "\n"
        "After evaluation, each tree becomes a **`Candidate`** that carries:\n"
        "- `fitness` \u2014 prediction error (lower is better)\n"
        "- `parsimony` \u2014 tree complexity (lower is better)\n"
        "- `tag` \u2014 evolution history\n"
        "\n"
        "**Tournament selection** picks parents: randomly sample `n` candidates, return\n"
        "the fittest one's tree.\n"
        "\n"
        "The **Pareto front** retains only *non-dominated* candidates:\n"
        "candidate A dominates B if `A.parsimony \u2264 B.parsimony` AND `A.fitness \u2264 B.fitness`\n"
        "(with at least one strict inequality).  This gives you a frontier of the\n"
        "accuracy-vs-complexity trade-off."
    )
)

cells.append(
    code(
        "# --- 5.1 Building a hand-crafted population ---\n"
        "from plagih.trees import Candidate\n"
        "\n"
        "# (tree, fitness, parsimony)\n"
        "pop_data = [\n"
        "    (make_tree_simple(),            0.8,  3),   # small, mediocre\n"
        "    (make_tree_trig(),              0.5,  5),   # medium, good\n"
        "    (make_tree_ifte(),              0.3,  5),   # medium, great\n"
        "    (make_tree_redundant(),         0.4, 11),   # large, good \u2014 dominated by ifte\n"
        "    (make_tree_crossover_parent_a(),0.6,  4),   # medium-small, ok\n"
        "    (make_tree_crossover_parent_b(),0.9,  3),   # small, bad\n"
        "]\n"
        "\n"
        "population = [Candidate(t, fitness=f, parsimony=p, tag='demo') for t, f, p in pop_data]\n"
        "\n"
        "for c in population:\n"
        "    print(c)"
    )
)

cells.append(
    code(
        "# --- 5.2 Tournament selection ---\n"
        "from plagih.trees import selection_tournament\n"
        "\n"
        "np.random.seed(3); import random; random.seed(3)\n"
        "winner_tree = selection_tournament(population, n=3)\n"
        "print('Tournament winner (tree):', winner_tree.get_sympy_expr())"
    )
)

cells.append(
    code(
        "# --- 5.3 Pareto front ---\n"
        "from plagih.paretofront import pareto_from_pop\n"
        "from plagih.visualization.tree_renderer import visualize_paretofront\n"
        "import tempfile, pathlib\n"
        "\n"
        "pareto = pareto_from_pop(population)\n"
        "print(f'Population size : {len(population)}')\n"
        "print(f'Pareto front    : {len(pareto)} candidates')\n"
        "for c in pareto:\n"
        "    print(f'  P={c.get_parsim()}, F={c.get_fitness():.1f}  \u2192  {c.get_evotree().get_sympy_expr()}')\n"
        "\n"
        "# Inline visualisation using a temporary directory\n"
        "tmp = pathlib.Path(tempfile.mkdtemp())\n"
        "visualize_paretofront(pareto, output_dir=tmp, dpi=100)\n"
        "img = plt.imread(str(list((tmp/'tree_output').glob('*.png'))[0]))\n"
        "fig, ax = plt.subplots(figsize=(min(10, 3*len(pareto)), 3))\n"
        "ax.imshow(img); ax.axis('off')\n"
        "plt.title('Pareto Front', fontweight='bold')\n"
        "plt.tight_layout(); plt.show()"
    )
)

# ── Part 6 ─────────────────────────────────────────────────────────────────────
cells.append(
    md(
        '<a id="part6"></a>\n'
        "## Part 6 — Monitoring\n"
        "\n"
        "`GPMonitor` collects per-generation statistics during a GP run.\n"
        "It supports arbitrary metric names, custom callbacks, and export to\n"
        "a pandas DataFrame for downstream analysis.\n"
        "\n"
        "Here we feed it hand-crafted generation data to demonstrate the API\n"
        "without running an actual evolution loop."
    )
)

cells.append(
    code(
        "# --- 6.1 GPMonitor: record & query ---\n"
        "# record_generation() auto-computes fit_best, parsim_best etc. from population.\n"
        "# extra_metrics can inject additional custom scalars.\n"
        "from plagih.monitoring import GPMonitor\n"
        "\n"
        "monitor = GPMonitor()\n"
        "\n"
        "# Simulate 5 generations of improvement (inject synthetic fitness trend)\n"
        "np.random.seed(42)\n"
        "for gen in range(5):\n"
        "    monitor.record_generation(\n"
        "        gen_id=gen,\n"
        "        population=population,\n"
        "        gen_time=0.3 + np.random.uniform(0, 0.1),\n"
        "        extra_metrics={\n"
        "            'demo_fitness': 0.8 - gen * 0.12 + np.random.uniform(-0.02, 0.02),\n"
        "            'demo_parsimony': 5 - gen * 0.4,\n"
        "        }\n"
        "    )\n"
        "\n"
        "df_monitor = monitor.to_dataframe()\n"
        "# gen_id is the index; reset_index() makes it a regular column for plotting\n"
        "df_monitor = df_monitor.reset_index()\n"
        "print('Monitor DataFrame (last 3 rows):')\n"
        "print(df_monitor[['gen_id', 'demo_fitness', 'demo_parsimony', 'time']].tail(3).to_string(index=False))"
    )
)

cells.append(
    code(
        "# --- 6.2 Performance plot ---\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3))\n"
        "\n"
        "if 'demo_fitness' in df_monitor.columns:\n"
        "    axes[0].plot(df_monitor['gen_id'], df_monitor['demo_fitness'], 'o-', color='#1565C0')\n"
        "    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Best fitness')\n"
        "    axes[0].set_title('Fitness over generations')\n"
        "\n"
        "if 'demo_parsimony' in df_monitor.columns:\n"
        "    axes[1].plot(df_monitor['gen_id'], df_monitor['demo_parsimony'], 's-', color='#B71C1C')\n"
        "    axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Best parsimony')\n"
        "    axes[1].set_title('Parsimony (complexity) over generations')\n"
        "\n"
        "plt.tight_layout(); plt.show()"
    )
)

# ── Part 7 ─────────────────────────────────────────────────────────────────────
cells.append(
    md(
        '<a id="part7"></a>\n'
        "## Part 7 — Targeted Optimization\n"
        "\n"
        "**Targeted optimization** analyses individual trees to guide mutations more\n"
        "intelligently than pure random search.\n"
        "\n"
        "### Phase 1 \u2014 Node intermediates\n"
        "`eval_node_intermediates(tree, df)` evaluates every node in a tree and returns\n"
        "the intermediate array at each node. This is the foundation for pseudo-backpropagation.\n"
        "\n"
        "### Phase 2 \u2014 Ifte pseudo-backpropagation\n"
        "`ifte_component_scores(tree, df, target)` analyses every `Ifte(cond, then, else)` in a tree:\n"
        "\n"
        "| Component | Scored as |\n"
        "|---|---|\n"
        "| `condition` | Fraction of rows where it selects the better branch |\n"
        "| `then` | Accuracy of the *then* value on rows where `cond=True` |\n"
        "| `else` | Accuracy of the *else* value on rows where `cond=False` |\n"
        "\n"
        "The **weakest** component is flagged for targeted mutation \u2014 this is what\n"
        "the `targeted_ifte` GP strategy exploits."
    )
)

cells.append(
    code(
        "# --- 7.1 Node intermediates ---\n"
        "from plagih.targeted_optimization import eval_node_intermediates\n"
        "\n"
        "df = make_sample_df()\n"
        "tree = make_tree_trig()   # sin(a*b + 2)\n"
        "intermediates = eval_node_intermediates(tree, df)\n"
        "\n"
        "print(f'Intermediate outputs at {len(intermediates)} nodes:')\n"
        "for node in tree.to_preorder():\n"
        "    if id(node) in intermediates:\n"
        "        vals = intermediates[id(node)]\n"
        "        print(f'  {type(node).__name__:10s}  \u2192  {np.round(vals, 3)}')"
    )
)

cells.append(
    code(
        "# --- 7.2 Ifte component scores ---\n"
        "# Tree: Ifte(cartPos < 0.5, cartVel, -1 * cartPos)\n"
        "# Target: the 'action' column (what the agent should output)\n"
        "from plagih.targeted_optimization import ifte_component_scores\n"
        "\n"
        "df_cp = make_cartpole_df()\n"
        "tree_ifte_cp = make_tree_ifte_cartpole()\n"
        "target = df_cp['action'].to_numpy()\n"
        "\n"
        "results = ifte_component_scores(tree_ifte_cp, df_cp, target)\n"
        "print(f'Ifte nodes analysed: {len(results)}')\n"
        "for r in results:\n"
        "    print(f'  Weakest component : {r.weakest}')\n"
        "    print(f'  Condition accuracy: {r.condition_accuracy:.2f}')\n"
        "    for name, sc in r.scores.items():\n"
        "        print(f'    {name:12s}  count_score={sc.count_score:.2f}  error_sum={sc.error_sum:.3f}  n_rows={sc.n_rows}')"
    )
)

cells.append(
    code(
        "# --- 7.3 Score-coloured tree render ---\n"
        "# Nodes are tinted from neutral (score=0, good) to red (score=1, weak).\n"
        "# The Ifte node itself is coloured by its weakest component's weakness.\n"
        "node_scores = make_ifte_node_scores(results)\n"
        "\n"
        "print('Ifte node scores (id \u2192 weakness):')\n"
        "for k, v in node_scores.items():\n"
        "    print(f'  node id {k}  \u2192  {v:.2f}')\n"
        "\n"
        "show_tree_with_scores(\n"
        "    tree_ifte_cp,\n"
        "    node_scores,\n"
        "    title=f'Ifte weakness map  (weakest: {results[0].weakest if results else \"N/A\"})'\n"
        ")"
    )
)

# ── Write ──────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0",
        },
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {OUT}  ({len(cells)} cells, {OUT.stat().st_size} bytes)")

nb2 = json.loads(OUT.read_text(encoding="utf-8"))
code_cells = [c for c in nb2["cells"] if c["cell_type"] == "code"]
md_cells = [c for c in nb2["cells"] if c["cell_type"] == "markdown"]
print(f"Verified: {len(nb2['cells'])} cells total, {len(code_cells)} code, {len(md_cells)} markdown")
