import pygraphviz as pgv


nlist = [0, 1, 2]
edges = [(0, 1), (0, 2)]
labels = {0: 'a',
          1: 'b',
          2: 'c'}

nodes2 = ['+', 'y']
edges2 = [('+', 'y')]
# labels2 = {4: 'x',
#            5: 'y'}
g = pgv.AGraph()
g.add_nodes_from(nlist)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nlist:
    n = g.get_node(i)
    n.attr["label"] = labels[i]


g.add_nodes_from(nodes2)
g.add_edges_from(edges2)
g.layout(prog="dot")

# for i in nodes2:
#     n = g.get_node(i)
#     n.attr["label"] = labels2[i]

g.draw("tree.pdf")
