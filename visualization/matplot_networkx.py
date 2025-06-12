import matplotlib.pyplot as plt
import networkx as nx
import sympy
from matplotlib.patches import Ellipse

from plagih.sympy_extras import plagih_sympify
from plagih.trees import (
    Node,
    Symbol, Number, Boolean, Terminal,
    BaseOperator, Add, Mul, Pow, Min, Max, sympy_to_tree  # ggf. alle wichtigen Operatoren importieren
)
import re

def format_float_label(val: float) -> str:
    """
    Kürzt Fließkommazahlen auf maximal 3 signifikante Nachkommastellen (ohne nachfolgende 0er).
    Beispiel: 2.300000000000000 → 2.3
              2.3456789 → 2.346
              0.00000012 → 0.000001 (kleinste 1-stellige signifikante)
    """
    # Sehr kleine Zahlen als 0.001 anzeigen
    if abs(val) < 0.001:
        return '0.001'
    # Formatieren mit max 3 Nachkommastellen, abschneiden
    return f"{val:.3f}".rstrip('0').rstrip('.')  # z. B. 2.300 → 2.3, 2.000 → 2

def extract_family_and_time(symbol: str):
    m = re.match(r'^(.+?)(?:_([+-]?\d+))?$', symbol)
    if m:
        fam = m.group(1)
        time = int(m.group(2)) if m.group(2) is not None else None
        return fam, time
    return symbol, None


COLOR_BY_CLASS = {
    BaseOperator: '#82b1ff',    # Operatoren hellblau
    Number: '#a5d6a7',          # Zahlen grün
    Symbol: '#fff59d',          # Variable (ggf. überschrieben bei Zeitinfo) gelb
    Boolean: '#ffe082',
    Terminal: '#e0e0e0',        # generische Blätter grau
}
OPERATOR_DISPLAY = {
    Add: '+',
    Mul: '⋅',
    Pow: 'x^y',
    Min: 'min',
    Max: 'max',
}

def get_node_color(node):
    # Spezialbehandlung für Symbol mit Zeitbezug
    if isinstance(node, Symbol):
        name = getattr(node, 'name', '')
        if '_' in name:
            val = node.get_value()
            fam, time, = extract_family_and_time(val)
            if time is not None:
                return '#ffe082' if str(time).startswith('+') else '#ffcc80'  # future vs. past
        return COLOR_BY_CLASS.get(Symbol)

    # Allgemeiner Lookup: Klasse matchen
    for cls, color in COLOR_BY_CLASS.items():
        if isinstance(node, cls):
            return color
    return '#e0e0e0'  # fallback


def get_display_label(node):
    from plagih.trees import Terminal

    # Operatoren erhalten ein Symbol, wenn definiert
    for cls, display in OPERATOR_DISPLAY.items():
        if isinstance(node, cls):
            return display

    # Terminale Knoten zeigen ihren Wert an
    if isinstance(node, Terminal):
        val = node.get_value()
        if isinstance(val, sympy.Number):
            return format_float_label(float(val))
        return str(val)

    else:
        return node.__class__.__name__

def classify_label(label):
    if label in {'add', 'mul', 'pow', 'min', 'max'}:
        return 'op'
    elif label.replace('.', '', 1).isdigit():
        return 'const'
    elif label.isalpha():
        return 'var'
    return 'other'

def draw_colored_elliptical_nodes(ax, pos, labels, node_lookup, width=1.2, height=0.6, fontsize=10):
    for nid, (x, y) in pos.items():
        node = node_lookup[nid]
        label = get_display_label(node)
        color = get_node_color(node)
        ellipse = Ellipse((x, y), width=width, height=height,
                          facecolor=color, edgecolor='black', zorder=1)
        ax.add_patch(ellipse)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, zorder=2)

def _hierarchy_absolute(G, level_spacing=1.0, horizontal_spacing=1.0, root=None):
    def dfs(node, depth=0):
        children = list(G.successors(node))
        if not children:
            x = dfs.counter * horizontal_spacing
            y = -depth * level_spacing
            pos[node] = (x, y)
            dfs.counter += 1
        else:
            for child in children:
                dfs(child, depth + 1)
            child_xs = [pos[c][0] for c in children]
            mid_x = sum(child_xs) / len(child_xs)
            y = -depth * level_spacing
            pos[node] = (mid_x, y)

    if root is None:
        root = [n for n, d in G.in_degree() if d == 0][0]

    pos = {}
    dfs.counter = 0
    dfs(root)
    return pos


def draw_tree_on_ax(ax, G, labels, pos, node_lookup, node_radius=0.3):
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], 'k-', linewidth=1, zorder=0)

    for nid, (x, y) in pos.items():
        node = node_lookup[nid]
        label = labels[nid]
        color = get_node_color(node)
        circle = plt.Circle((x, y), radius=node_radius, facecolor=color, edgecolor='black', zorder=1)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, zorder=2)


def build_tree_graph_from_nodes(node: Node, offset_x=0.0, offset_y=0.0, start_id=0,
                                level_spacing=1.0, horizontal_spacing=1.0):
    G = nx.DiGraph()
    labels = {}
    node_lookup = {}
    counter = [start_id]

    def build(n: Node, parent=None, depth=0):
        nid = counter[0]
        counter[0] += 1

        G.add_node(nid)
        labels[nid] = get_display_label(n)
        node_lookup[nid] = n

        if parent is not None:
            G.add_edge(parent, nid)

        if hasattr(n, 'has_childs') and n.has_childs():
            for child in n.get_childs():
                build(child, parent=nid, depth=depth + 1)

    build(node)

    pos = _hierarchy_absolute(G, level_spacing=level_spacing, horizontal_spacing=horizontal_spacing)
    for k in pos:
        x, y = pos[k]
        pos[k] = (x + offset_x, y + offset_y)

    return G, labels, pos, node_lookup



def plot_tree_nodewise(node: Node, level_spacing=1.0, horizontal_spacing=1.0, node_radius=0.3):
    G, labels, pos, node_lookup = build_tree_graph_from_nodes(
        node, level_spacing=level_spacing, horizontal_spacing=horizontal_spacing
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    # Kanten
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], 'k-', linewidth=1, zorder=0)

    # Knoten
    for nid, (x, y) in pos.items():
        n = node_lookup[nid]
        label = labels[nid]
        color = get_node_color(n)
        circle = plt.Circle((x, y), radius=node_radius, facecolor=color, edgecolor='black', zorder=1)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, zorder=2)

    ax.set_xlim(min(x for x, y in pos.values()) - 1, max(x for x, y in pos.values()) + 1)
    ax.set_ylim(min(y for x, y in pos.values()) - 1, max(y for x, y in pos.values()) + 1)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.show()

def plot_trees_nodewise(tree_list, spacing=6.0, level_spacing=1.0, horizontal_spacing=1.0, node_radius=0.3):
    fig, ax = plt.subplots(figsize=(spacing * len(tree_list), 4))

    offset = 0.0
    total_nodes = 0

    for tree in tree_list:
        G, labels, pos, node_lookup = build_tree_graph_from_nodes(
            tree,
            offset_x=offset,
            start_id=total_nodes,
            level_spacing=level_spacing,
            horizontal_spacing=horizontal_spacing
        )

        # Kanten
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], 'k-', linewidth=1, zorder=0)

        # Knoten
        for nid, (x, y) in pos.items():
            n = node_lookup[nid]
            label = labels[nid]
            color = get_node_color(n)
            circle = plt.Circle((x, y), radius=node_radius, facecolor=color, edgecolor='black', zorder=1)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', fontsize=10, zorder=2)

        offset += spacing
        total_nodes += len(G.nodes)

    ax.axis('off')
    ax.set_aspect('equal')
    plt.show()



if __name__ == "__main__":
    syex = plagih_sympify('a + 2.3')
    t = sympy_to_tree(syex, allow_chain=True)
    plot_tree_nodewise(t)

    tree1 = ['add', ['pow', ['x']], ['mul', ['2'], ['y']]]
    tree2 = ['min', ['x'], ['max', ['y'], ['3']]]
    sy1 = plagih_sympify('a + 2.3')
    sy2 = plagih_sympify('min(x, max(y, 3))')
    sy3 = plagih_sympify('sqrt(x**2 + y**2)')

    trees = [sympy_to_tree(s, allow_chain=True) for s in [sy1, sy2, sy3]]
    plot_trees_nodewise(trees)
    extract_family_and_time("cartVel")       # → ('cartVel', None)
    extract_family_and_time("cartVel_2")     # → ('cartVel', 2)
    extract_family_and_time("pos_-1")        # → ('pos', -1)
    extract_family_and_time("angle_+3")      # → ('angle', 3)