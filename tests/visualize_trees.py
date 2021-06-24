import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import networkx as nx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def caltulate_grid_granularity(tree):
    pass


def label(xy, text):
    y = xy[1] - 0.01  # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=12)
    return


def add_circle(level, pos_x, name):
    grid_num = get_grid(level, pos_x)
    circle = mpatches.Circle(grid[grid_num], 0.05, ec="none")
    label(grid[grid_num], str(name))
    patches.append(circle)
    return


def add_box(level, pos_x):
    # add a fancy box
    grid_num = get_grid(level, pos_x)
    box_width = 0.1
    box_height = 0.05
    fancybox = mpatches.FancyBboxPatch(
        grid[grid_num] - [box_width / 2, box_height / 2], box_width, box_height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02))
    label(grid[grid_num], "Ifte")
    patches.append(fancybox)
    return


def add_link(aa, bb):
    # add a line
    a = grid[get_grid(aa[0], aa[1])]
    b = grid[get_grid(bb[0], bb[1])]
    x, y = np.array([[a[0], b[0]], [a[1], b[1]]])
    line = mlines.Line2D(x, y, lw=2., alpha=0.9)
    ax.add_line(line)
    return line


def get_grid(lvl, pos_x):

    num = lvl * dim_x + pos_x
    return num


def make_grid(dim_x, dim_y):
    grid = np.mgrid[0.1:0.9:dim_x * 1j, 0.9:0.1:-dim_y * 1j].T
    grid = grid.reshape(dim_y * dim_x, 2)
    return grid


def auto_enum():
    for i in range(0, len(grid)):
        label(grid[i], str(i))


def tree_graph(expr):
    """
    Construct the graph of a tree expression. The tree expression must be
    valid. It returns in order a node list, an edge list, and a dictionary of
    the per node labels. The node are represented by numbers, the edges are
    tuples connecting two nodes (number), and the labels are values of a
    dictionary for which keys are the node numbers.
    :param expr: A tree expression to convert into a graph.
    :returns: A node list, an edge list, and a dictionary of labels.
    The returned objects can be used directly to populate a
    `NetworX <http://networkx.github.com/>`_ graph::
        import matplotlib.pyplot as plt
        import networkx as nx
        # [...] Execution of code that produce a tree expression
        nodes, edges, labels = graph(nlabel)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.graphviz_layout(g, prog="dot")
        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()
    .. note::
       We encourage you to use `pygraphviz
       <http://networkx.lanl.gov/pygraphviz/>`_ as the nodes might be plotted
       out of order when using `NetworX <http://networkx.github.com/>`_.
    """
    nodes = range(len(expr))
    edges = list()
    labels = dict()

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = node.name if isinstance(node, Primitive) else node.value
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels

def test1():
    fig, ax = plt.subplots()
    dim_x = 7
    dim_y = 3
    grid = make_grid(dim_x, dim_y)

    patches = []

    add_box(0, 3)

    add_circle(1, 1, '<')
    add_circle(1, 3, '0')
    add_circle(1, 5, '2')
    add_circle(2, 0, 'obs')
    add_circle(2, 2, '0')

    colors = np.linspace(1, 0, 2)
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.9)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    add_link((0, 3), (1, 1))
    add_link((0, 3), (1, 3))
    add_link((0, 3), (1, 5))
    add_link((1, 1), (2, 0))
    add_link((1, 1), (2, 2))

    plt.axis('off')
    plt.axis('scaled')

    plt.tight_layout()

    plt.show()



