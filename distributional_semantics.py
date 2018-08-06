import gensim
import os
import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout
from chinese_whispers import chinese_whispers, aggregate_clusters
from gensim.models.poincare import PoincareModel


def process_input(taxonomy):
    """ Read the taxonomy and generate a networkx graph """

    # Read the taxonomy as a dataframe
    print('Reading input...')
    df = pd.read_csv(
        taxonomy,
        sep='\t',
        header=None,
        names=['hyponym', 'hypernym'],
        usecols=[1, 2],
    )

    # Construct the networkx graph
    print('Constructing networkx graph...')
    graph = nx.DiGraph()
    for hypernym, hyponym in zip(list(df['hypernym']), list(df['hyponym'])):
        # Simplify the compound words by replacing the whitespaces with underscores
        if ' ' in hypernym:
            hypernym = '_'.join(hypernym.split())
        if ' ' in hyponym:
            hyponym = '_'.join(hyponym.split())
        graph.add_edge(hypernym, hyponym)

    return graph


def display_taxonomy(graph):
    """ Display the taxonomy in a hierarchical layout """

    pos = graphviz_layout(graph, prog='dot', args="-Grankdir=LR")
    plt.figure(3, figsize=(48, 144))
    nx.draw(graph, pos, with_labels=True, arrows=True)
    plt.show()


def load_vectors(embedding):
    """ Load word vectors. """

    embedding_dir = '/home/5aly/taxi/distributed_semantics/embeddings/'

    if embedding == "wiki2M":
        model = gensim.models.KeyedVectors.load_word2vec_format(embedding_dir + 'crawl-300d-2M.vec', binary=False)
    elif embedding == "wiki1M_subword":
        model = gensim.models.KeyedVectors.load_word2vec_format(
            embedding_dir + 'wiki-news-300d-1M-subword.vec', binary=False
        )
    elif embedding == "own_w2v":
        model = gensim.models.KeyedVectors.load(embedding_dir + 'own_embeddings_w2v')
    elif embedding == "poincare":
        model = PoincareModel.load(embedding_dir + 'embeddings_poincare_wordnet')

    return model


def create_children_clusters(w2v_model, graph, embedding, depth):
    """ This function returns a dictionary where corresponding to each key(node) is a graph of its children """

    clustered_graph = {}
    for node in graph.nodes():
        clustered_graph[node] = nx.Graph()
        successors = [s.lower() for s in graph.successors(node)]

        for successor in successors:
            try:
                if embedding == "poincare":
                    for word, score in w2v_model.kv.most_similar(successor, topn=depth):
                        if word.lower() in successors:
                            clustered_graph[node].add_edge(successor, word.lower())
                else:
                    for word, score in w2v_model.most_similar(successor, topn=depth):
                        if word.lower() in successors:
                            clustered_graph[node].add_edge(successor, word.lower())
            except KeyError:
                successor_terms = successor.split('_')
                root_terms = [successor_terms[0], successor_terms[-1]]
                if node in root_terms:
                    clustered_graph[node].add_node(successor)

    return clustered_graph


def calculate_similarity(w2v_model, parent, family, cluster, embedding):
    # Similarity between the parent and a cluster
    parent_similarity = 0
    for item in cluster:
        try:
            if embedding == "poincare":
                parent_similarity += w2v_model.kv.similarity(parent, item)
            else:
                parent_similarity += w2v_model.similarity(parent, item)
        except KeyError:  # skip the terms not in vocabulary
            continue
    parent_similarity /= len(cluster)

    # Similarity between a family and a cluster
    family_similarity = 0
    for f_item in family:
        for c_item in cluster:
            try:
                if embedding == "poincare":
                    family_similarity += w2v_model.kv.simsimilarity(f_item, c_item)
                else:
                    family_similarity += w2v_model.similarity(f_item, c_item)
            except KeyError:  # skip the terms not in vocabulary
                continue
    family_similarity /= (len(family) * len(cluster))

    # Final score is the average of both the similarities
    return (parent_similarity + family_similarity) / 2


def tune_result(g_improved):
    """ Filter the results i.e. remove all the isolated nodes and nodes with blank labels """

    print('Tuning the result...')

    if '' in g_improved.nodes():
        g_improved.remove_node('')

    hypernyms = {x[0] for x in g_improved.edges()}
    isolated_nodes = list(nx.isolates(g_improved))
    for isolated_node in isolated_nodes:
        terms = isolated_node.split('_')
        if terms[-1] in hypernyms:
            g_improved.add_edge(terms[-1], isolated_node)
        elif terms[0] in hypernyms:
            g_improved.add_edge(terms[0], isolated_node)
        else:
            g_improved.remove_node(isolated_node)

    return g_improved


def apply_distributional_semantics(nx_graph, mode, embeddings, depth, iterations):
    # Load the pre-trained vectors
    print('Loading', embeddings, 'embeddings...')
    w2v_model = load_vectors(embeddings)

    g_improved = nx_graph.copy()
    for i in range(iterations):
        print('\n\nIteration %d/%d:' % (i, iterations))

        # Remove small clusters
        print('Removing small clusters..')
        g_clustered = create_children_clusters(w2v_model, g_improved, embeddings, depth)
        removed_clusters = []

        for node, graph in g_clustered.items():
            gc = chinese_whispers(graph, weighting='top', iterations=60)
            try:
                max_cluster_size = len(max(aggregate_clusters(gc).values(), key=len))
            except ValueError:
                continue
            for label, cluster in aggregate_clusters(gc).items():  # detach all the clusters smaller than the maximum
                if len(cluster) < max_cluster_size:
                    removed_clusters.append(cluster)
                    for item in cluster:
                        g_improved.remove_edge(node, item)

        if mode == 'reattach':  # Reattach the removed clusters
            print('Reattaching removed clusters...')
            g_detached = create_children_clusters(w2v_model, g_improved, embeddings, depth)
            for cluster in removed_clusters:
                max_score = 0
                max_score_node = ''
                for node, graph in g_detached.items():
                    gc = chinese_whispers(graph, weighting='top', iterations=60)
                    for label, family in aggregate_clusters(gc).items():
                        score = calculate_similarity(w2v_model, node, family, cluster, embeddings)
                        if score > max_score:
                            max_score = score
                            max_score_node = node
                for item in cluster:
                    g_improved.add_edge(max_score_node, item)

        # Tune the result
        g_improved = tune_result(g_improved)

    return g_improved


def save_result(result, path, mode):
    print('Saving the result...')
    df_improved = pd.DataFrame(list(result.edges()), columns=['hypernym', 'hyponym'])
    df_improved = df_improved[df_improved.columns.tolist()[::-1]]

    # Replace the underscores with blanks
    df_improved['hyponym'] = df_improved['hyponym'].apply(lambda x: x.replace('_', ' '))
    df_improved['hypernym'] = df_improved['hypernym'].apply(lambda x: x.replace('_', ' '))

    result_path = os.path.splitext(path)
    output_path = 'taxi_output/distributional_semantics/' + str(result_path[0].split('/')[-1]) + '-semantic'
    if mode == 'only_removal':
        output_path += '-removal'
    output_path += result_path[1]

    df_improved.to_csv(output_path, sep='\t', header=False)
    print('Output saved at:', output_path)


def main(taxonomy, mode, embeddings, depth, iterations):

    # Read the input
    graph = process_input(taxonomy)

    # Distributional Semantics
    g_improved = apply_distributional_semantics(graph, mode, embeddings, depth, iterations)

    # Save the results
    save_result(g_improved, taxonomy, mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributional Semantics for Taxonomy')
    parser.add_argument('-t', '--taxonomy', required=True, help='Input file containing the taxonomy')
    parser.add_argument(
        '-e', '--embeddings', required=True, type=str,
        choices=['poincare', 'wiki2M', 'wiki1M_subword', 'own_w2v'],
        help='Classifier architecture of the system.'
    )
    parser.add_argument('-m', '--mode', default='reattach', choices=['only_removal', 'reattach'])
    parser.add_argument(
        '-d', '--depth', type=int, default=100,
        help='Number of results to return while checking for most similar nodes of a term.'
    )
    parser.add_argument('-i', '--iterations', type=int, default=1, help='Number of iterations.')
    args = parser.parse_args()

    print('Input File:', args.taxonomy)
    print('Embeddings:', args.embeddings)
    print('Mode:', args.mode)

    main(args.taxonomy, args.mode, args.embeddings, args.depth, args.iterations)
