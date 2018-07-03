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
for rel in zip(list(df['hypernym']), list(df['hyponym'])):
    G.add_edge(rel[0].decode('utf-8'), rel[1].decode('utf-8'))

image_path = os.path.join('networkx_visualizations', os.path.splitext(sys.argv[1])[0].split('/')[-1])

# Convert the graph into a tree structure
pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")

# Increase the image size for better visualization
plt.figure(3, figsize=(48, 144))

# Save the graph with labelled nodes
nx.draw(G, pos, with_labels=True, arrows=True)
plt.savefig(image_path + '.png')
