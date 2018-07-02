



def remove_edges_from_network_graph(g, edges_to_remove):
    for edge in edges_to_remove:
        print("Remove edge: (%s, %s)" % edge)
        g.remove_edge(edge[0], edge[1])

    print("Edges removed: %s" % len(edges_to_remove))
