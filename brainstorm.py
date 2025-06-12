from graphviz import Digraph
from PIL import Image
import io

def show_tree_inline(tree):
    graph = Digraph(format='png')
    graph.attr('node', shape='box', style='filled', color='lightgrey', fontname='Helvetica')
    node_counter = [0]

    def recurse(subtree, parent_id=None):
        node_id = f'n{node_counter[0]}'
        node_counter[0] += 1

        if isinstance(subtree, list) and isinstance(subtree[0], str):
            graph.node(node_id, subtree[0])
            if parent_id:
                graph.edge(parent_id, node_id)
            for child in subtree[1:]:
                recurse(child, parent_id=node_id)
        else:
            graph.node(node_id, str(subtree))
            if parent_id:
                graph.edge(parent_id, node_id)

    recurse(tree)

    # direkt in Bytes rendern, nicht speichern
    png_bytes = graph.pipe()
    image = Image.open(io.BytesIO(png_bytes))
    image.show()

# Beispiel
tree = ['Max', ['Pow', ['Mul', ['2'], ['Round']]]]
show_tree_inline(tree)