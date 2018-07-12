import os
import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout


ap = argparse.ArgumentParser()
ap.add_argument('-t', '--taxonomy', required=True, help='Taxonomy file path')
args = vars(ap.parse_args())


# Read the CSV (Tab separated) file
df = pd.read_csv(
    args['taxonomy'],
    sep='\t',
    header=None,
    names=['hyponym', 'hypernym'],
    usecols=[1, 2]
)


# Generate a Directed Graph
G = nx.DiGraph()
for rel in zip(list(df['hypernym']), list(df['hyponym'])):
    G.add_edge(rel[0], rel[1])

image_path = os.path.join(os.getcwd(), os.path.splitext(args['taxonomy'])[0].split('/')[-1])

# Convert the graph into a tree structure
pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")

# Increase the image size for better visualization
plt.figure(3, figsize=(48, 144))

# Save the graph with labelled nodes
nx.draw(G, pos, with_labels=True, arrows=True)
plt.savefig(image_path + '.png')
