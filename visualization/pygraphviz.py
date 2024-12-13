from plagih.trees import Node


def render_pygraphviz(tree: Node):
    import pygraphviz as pgv
    node_dict, edges = tree.get_all_nodes_visualize('0')
    vnodes = node_dict.keys()
    vlabels = {}
    for k, v in node_dict.items():
        vlabels[k] = v['showme']

    g = pgv.AGraph()
    g.add_nodes_from(vnodes)
    g.add_edges_from(edges)

    for i in vnodes:
        n = g.get_node(i)
        n.attr["label"] = vlabels[i]
    g.layout(prog="dot")

    # for i in nodes2:
    #     n = g.get_node(i)
    #     n.attr["label"] = labels2[i]

    g.draw("tree.png")
