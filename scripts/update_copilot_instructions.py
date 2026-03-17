"""
Auto-update copilot-instructions.md and ARCHITECTURE.md from source code.

Scans module docstrings and the Node class hierarchy in ``plagih/trees/_nodes.py``
and replaces sections delimited by AUTOGEN markers in both Markdown files.

Run manually or as a pre-commit hook:

    python scripts/update_copilot_instructions.py

Exit code 1 = files were changed (pre-commit will re-stage & retry).
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTRUCTIONS_PATH = PROJECT_ROOT / ".github" / "copilot-instructions.md"
ARCHITECTURE_PATH = PROJECT_ROOT / "docs" / "ARCHITECTURE.md"
TREES_PATH = PROJECT_ROOT / "plagih" / "trees" / "_nodes.py"

# Modules to scan: (path relative to project root, display name)
MODULES = [
    ("plagih/trees/__init__.py", "plagih/trees/"),
    ("plagih/trees/_nodes.py", "plagih/trees/_nodes.py"),
    ("plagih/trees/_evolution.py", "plagih/trees/_evolution.py"),
    ("plagih/trees/_gp_engine.py", "plagih/trees/_gp_engine.py"),
    ("plagih/parallel.py", "plagih/parallel.py"),
    ("plagih/paretofront.py", "plagih/paretofront.py"),
    ("plagih/monitoring.py", "plagih/monitoring.py"),
    ("plagih/evaluation_context.py", "plagih/evaluation_context.py"),
    ("plagih/population_merge.py", "plagih/population_merge.py"),
    ("plagih/util.py", "plagih/util.py"),
    ("visualization/tree_renderer.py", "visualization/tree_renderer.py"),
    ("visualization/visualize_trees.py", "visualization/visualize_trees.py"),
]

# Mixins - classes in trees.py that are NOT part of the Node hierarchy
# but are used as secondary bases via multiple inheritance.
MIXIN_CLASSES = {"ChainableOp", "CustomOperator", "NoSymCapitalized", "PleaseUsePartnerOp"}

# Classes to ignore entirely (not Node, not a mixin, just helpers)
IGNORE_CLASSES = {"RoundDummy", "_ClipAutocast"}

# Abstract / intermediate classes whose children should be grouped in compact view
ABSTRACT_CLASSES = {
    "Node",
    "NodeWithChilds",
    "NodeDummy",
    "BaseOperator",
    "MathOperator",
    "LogicOperator",
    "RelationalOperator",
    "Trigonometry",
    "BaseMinMax",
    "Terminal",
}


# =============================================================================
# Helpers
# =============================================================================


def get_first_docstring_line(filepath: Path) -> str:
    """Extract the first meaningful line of a module's docstring."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        if docstring:
            for line in docstring.strip().splitlines():
                line = line.strip()
                if line and not line.startswith("="):
                    return line
    except Exception:
        pass
    return "*(no docstring)*"


def count_classes_and_functions(filepath: Path) -> tuple:
    """Count top-level classes and functions in a module."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = sum(1 for node in ast.iter_child_nodes(tree) if isinstance(node, ast.ClassDef))
        functions = sum(1 for node in ast.iter_child_nodes(tree) if isinstance(node, ast.FunctionDef))
        return classes, functions
    except Exception:
        return 0, 0


def replace_autogen_section(content: str, marker: str, new_body: str) -> str:
    """Replace content between ``<!-- AUTOGEN:{marker}:START -->`` and ``END`` markers.

    Returns the updated string.  If markers are not found, returns *content* unchanged.
    """
    start_tag = f"<!-- AUTOGEN:{marker}:START -->"
    end_tag = f"<!-- AUTOGEN:{marker}:END -->"
    pattern = re.compile(
        re.escape(start_tag) + r".*?" + re.escape(end_tag),
        re.DOTALL,
    )
    replacement = f"{start_tag}\n{new_body}\n{end_tag}"
    return pattern.sub(replacement, content)


# =============================================================================
# Module Map
# =============================================================================


def build_module_table() -> str:
    """Build the markdown module map table."""
    lines = ["| Module | Responsibility |", "|---|---|"]
    for rel_path, display in MODULES:
        filepath = PROJECT_ROOT / rel_path
        if filepath.exists():
            desc = get_first_docstring_line(filepath)
            classes, funcs = count_classes_and_functions(filepath)
            suffix = f" ({classes}C/{funcs}F)" if classes or funcs else ""
            lines.append(f"| `{display}` | {desc}{suffix} |")
        else:
            lines.append(f"| `{display}` | *(file not found)* |")
    return "\n".join(lines)


# =============================================================================
# Node Hierarchy Parser
# =============================================================================


def _base_names(class_def: ast.ClassDef) -> List[str]:
    """Return base class names from an AST ClassDef node."""
    names = []
    for base in class_def.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def parse_node_hierarchy(
    filepath: Path,
) -> Tuple[
    Dict[str, List[str]],  # children: parent -> [child, ...]
    Dict[str, List[str]],  # bases_map: class -> [base, ...]
    Set[str],  # all_node_classes (transitively inheriting Node)
    Set[str],  # mixin_classes found
]:
    """Parse ``trees/_nodes.py`` and extract the Node class hierarchy via AST.

    Returns:
        children: mapping parent_class -> list of direct child classes
        bases_map: mapping class -> list of all base class names
        all_node_classes: set of classes that transitively inherit from Node
        mixin_classes: set of mixin/marker classes (not inheriting Node)
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Collect all top-level classes with their bases
    class_bases: Dict[str, List[str]] = {}
    class_order: List[str] = []  # preserve source order
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if node.name in IGNORE_CLASSES:
                continue
            class_bases[node.name] = _base_names(node)
            class_order.append(node.name)

    # Determine which classes transitively inherit from Node
    node_classes: Set[str] = {"Node"}

    changed = True
    while changed:
        changed = False
        for cls, bases in class_bases.items():
            if cls not in node_classes:
                for b in bases:
                    if b in node_classes:
                        node_classes.add(cls)
                        changed = True
                        break

    # Build children map (primary inheritance = first Node-inheriting base)
    children: Dict[str, List[str]] = defaultdict(list)
    for cls in class_order:
        if cls not in node_classes or cls == "Node":
            continue
        bases = class_bases[cls]
        # Find primary parent: first base that is a Node class
        primary = None
        for b in bases:
            if b in node_classes:
                primary = b
                break
        if primary:
            children[primary].append(cls)

    # Identify mixins actually used
    found_mixins: Set[str] = set()
    for cls in class_order:
        if cls in MIXIN_CLASSES:
            found_mixins.add(cls)

    return dict(children), class_bases, node_classes, found_mixins


def _get_mixins(cls_name: str, bases_map: Dict[str, List[str]], node_classes: Set[str]) -> List[str]:
    """Return non-Node base classes (mixins) for a given class."""
    bases = bases_map.get(cls_name, [])
    return [b for b in bases if b not in node_classes and b != "ABC" and b not in IGNORE_CLASSES]


def _render_tree(
    root: str,
    children: Dict[str, List[str]],
    bases_map: Dict[str, List[str]],
    node_classes: Set[str],
    prefix: str = "",
    is_last: bool = True,
    compact: bool = False,
) -> List[str]:
    """Recursively render a class hierarchy as an ASCII tree.

    Args:
        compact: If True, collapse leaf groups into a single line.
    """
    lines: List[str] = []
    connector = "└── " if is_last else "├── "
    mixins = _get_mixins(root, bases_map, node_classes)
    mixin_str = f"  (+ {', '.join(mixins)})" if mixins else ""

    kids = children.get(root, [])

    lines.append(f"{prefix}{connector}{root}{mixin_str}")

    child_prefix = prefix + ("    " if is_last else "│   ")

    if compact and kids:
        # Separate abstract children (have own sub-children) from leaf children
        abstract_kids = [k for k in kids if k in ABSTRACT_CLASSES or k in children]
        leaf_kids = [k for k in kids if k not in ABSTRACT_CLASSES and k not in children]

        all_display = abstract_kids + (["_LEAVES_"] if leaf_kids else [])
        for i, kid in enumerate(all_display):
            is_last_kid = i == len(all_display) - 1
            if kid == "_LEAVES_":
                # Collapse leaf classes into one line
                leaf_strs = []
                for lk in leaf_kids:
                    mx = _get_mixins(lk, bases_map, node_classes)
                    mx_s = f" (+{''.join(m[0] for m in mx)})" if mx else ""
                    leaf_strs.append(f"{lk}{mx_s}")
                leaf_connector = "└── " if is_last_kid else "├── "
                lines.append(f"{child_prefix}{leaf_connector}{', '.join(leaf_strs)}")
            else:
                lines.extend(
                    _render_tree(
                        kid,
                        children,
                        bases_map,
                        node_classes,
                        child_prefix,
                        is_last_kid,
                        compact=True,
                    )
                )
    else:
        for i, kid in enumerate(kids):
            is_last_kid = i == len(kids) - 1
            lines.extend(
                _render_tree(
                    kid,
                    children,
                    bases_map,
                    node_classes,
                    child_prefix,
                    is_last_kid,
                    compact=compact,
                )
            )

    return lines


def build_hierarchy_compact() -> str:
    """Build a compact Node hierarchy for copilot-instructions.md.

    Shows abstract/intermediate classes explicitly and collapses
    concrete leaf operators into comma-separated groups.
    """
    children, bases_map, node_classes, mixins = parse_node_hierarchy(TREES_PATH)
    lines = ["```"]
    tree_lines = _render_tree("Node", children, bases_map, node_classes, compact=True)
    # Replace root connector with plain name
    tree_lines[0] = "Node (ABC)"
    lines.extend(tree_lines)
    if mixins:
        lines.append("")
        lines.append(f"Mixins: {', '.join(sorted(mixins))}")
    lines.append("```")
    return "\n".join(lines)


def build_hierarchy_full() -> str:
    """Build a full Node hierarchy for ARCHITECTURE.md.

    Shows every concrete class on its own line with mixin annotations.
    """
    children, bases_map, node_classes, mixins = parse_node_hierarchy(TREES_PATH)
    lines = ["```"]
    tree_lines = _render_tree("Node", children, bases_map, node_classes, compact=False)
    tree_lines[0] = "Node (ABC)"
    lines.extend(tree_lines)
    if mixins:
        lines.append("")
        lines.append("Mixins (secondary bases, not part of Node tree):")
        for m in sorted(mixins):
            lines.append(f"  - {m}")
    lines.append("```")
    return "\n".join(lines)


# =============================================================================
# Main Update Logic
# =============================================================================


def update_file(filepath: Path, sections: Dict[str, str]) -> bool:
    """Update AUTOGEN sections in a markdown file.

    Args:
        filepath: Path to the markdown file.
        sections: Dict of marker_name -> new_body.

    Returns:
        True if file was modified.
    """
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return False

    content = filepath.read_text(encoding="utf-8")
    new_content = content

    for marker, body in sections.items():
        new_content = replace_autogen_section(new_content, marker, body)

    if new_content != content:
        filepath.write_text(new_content, encoding="utf-8")
        print(f"  Updated {filepath.name}: {', '.join(sections.keys())}")
        return True
    return False


def update_instructions():
    """Update all AUTOGEN sections in copilot-instructions.md and ARCHITECTURE.md."""
    changed = False

    module_table = build_module_table()
    hierarchy_compact = build_hierarchy_compact()
    hierarchy_full = build_hierarchy_full()

    # copilot-instructions.md
    changed |= update_file(
        INSTRUCTIONS_PATH,
        {
            "MODULE_MAP": module_table,
            "NODE_HIERARCHY_COMPACT": hierarchy_compact,
        },
    )

    # ARCHITECTURE.md
    changed |= update_file(
        ARCHITECTURE_PATH,
        {
            "NODE_HIERARCHY_FULL": hierarchy_full,
        },
    )

    if changed:
        print("Auto-generated doc sections were updated. Please re-stage the changed files.")
        sys.exit(1)
    else:
        print("No changes needed.")


if __name__ == "__main__":
    update_instructions()
