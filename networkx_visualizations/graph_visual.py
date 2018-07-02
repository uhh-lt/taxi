import sys
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import write_dot, graphviz_layout


# Read the CSV (Tab separated) file
df = pd.read_csv(
    sys.argv[1],
    sep='\t',
    header=None,
    names=['hyponym', 'hypernym'],
    usecols=[1, 2]
)


# Generate a Directed Graph
G = nx.DiGraph()
for rel in zip(list(df['hyponym']), list(df['hypernym'])):
    G.add_edge(rel[0].decode('utf-8'), rel[1].decode('utf-8'))

image_path = os.path.join('networkx_visualizations', os.path.splitext(sys.argv[1])[0].split('/')[-1])

# Convert the graph into a tree structure
pos = graphviz_layout(G, prog='dot')

# Save the graph with un-labelled nodes
nx.draw(G, pos, with_labels=False, arrows=True)
plt.savefig(image_path + '.png')

# Save the graph with labelled nodes
nx.draw(G, pos, with_labels=True, arrows=True)
plt.savefig(image_path + '_labelled.png')
