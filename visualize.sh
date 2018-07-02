#!/usr/bin/env bash

echo "Constructing a hierarchical structure of the taxonomy..."
python networkx_visualizations/graph_visual.py $1
echo "Done."
echo "The structure images are saved in the networkx_visualization directory."
