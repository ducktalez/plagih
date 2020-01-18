import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_nodes_from(['a', 'a', 'a', 'c'])
G.add_edges_from([('a', 'a'), ('a', 'b'), ('a', 'c')])

print(G.number_of_nodes())

nx.draw_networkx(G, with_labels=True, font_weight='bold')
plt.axis('off')
plt.show()