import gensim
import os
import argparse
import subprocess
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout
from chinese_whispers import chinese_whispers, aggregate_clusters
from gensim.models.poincare import PoincareModel
from nltk.corpus import wordnet as wn


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


def load_vectors():
    """ Load word vectors. """

    embedding_dir = '/home/5aly/taxi/distributed_semantics/embeddings/'

    poincare_model = model = PoincareModel.load(embedding_dir + 'embeddings_poincare_wordnet')  # parent-cluster relationship
    own_model = gensim.models.KeyedVectors.load(embedding_dir + 'own_embeddings_w2v')  # family-cluster relationship

    return poincare_model, own_model


def create_children_clusters(own_model, graph, depth):
    """ This function returns a dictionary where corresponding to each key(node) is a graph of its children """
    clustered_graph = {}
    for node in graph.nodes():
        clustered_graph[node] = nx.Graph()
        successors = [s.lower() for s in graph.successors(node)]

        for successor in successors:
            try:
                for word, score in own_model.most_similar(successor, topn=depth):
                    if word.lower() in successors:
                        clustered_graph[node].add_edge(successor, word.lower())
            except KeyError:  # If the word in not in vocabulary, check using the substring based method
                successor_terms = successor.split('_')
                root_terms = [successor_terms[0], successor_terms[-1]]
                if node in root_terms:
                    clustered_graph[node].add_node(successor)
    
    return clustered_graph


def remove_clusters(own_model, nx_graph, clusters_touched, depth, buffer):
    """ Removes the less related and small clusters from the graph """

    print('Removing small clusters..')
    g_clustered = create_children_clusters(own_model, nx_graph, depth)
    removed_clusters = []

    nodes, clusters, size_ratio = [], [], []
    for node, graph in g_clustered.items():
        gc = chinese_whispers(graph, weighting='top', iterations=60)
        try:  # Get the length of the largest cluster
            max_cluster_size = len(max(aggregate_clusters(gc).values(), key=len))
        except ValueError:
            continue
        
        # Calculate the size ratio of all the clusters which are smaller than the largest
        for _, cluster in aggregate_clusters(gc).items():
            if len(cluster) < max_cluster_size and cluster not in clusters_touched:
                nodes.append(node)
                clusters.append(cluster)
                size_ratio.append(len(cluster) / max_cluster_size)
    
    # Sort the small clusters according to their size_ratio
    sorted_node_clusters = [(node, cluster) for _, cluster, node in sorted(zip(size_ratio, clusters, nodes))]
    if len(sorted_node_clusters) > buffer:
        sorted_node_clusters = sorted_node_clusters[:buffer]

    for node, cluster in sorted_node_clusters:  # detach only the smallest 10 clusters in the entire taxonomy
        removed_clusters.append(cluster)
        for item in cluster:
            nx_graph.remove_edge(node, item)

    return nx_graph, removed_clusters


def calculate_similarity(poincare_model, own_model, parent, family, cluster, exclude_parent, exclude_family):
    
    # Similarity between the parent and a cluster
    parent_similarity = 0
    if not exclude_parent:
        parent_similarities = []
        for item in cluster:
            max_similarity = 0
            item_senses = wn.synsets(item)
            parent_senses = wn.synsets(parent)
            for parent_sense in parent_senses:
                for item_sense in item_senses:
                    try:
                        similarity = poincare_model.kv.similarity(parent_sense.name(), item_sense.name())
                        if similarity > max_similarity:
                            max_similarity = similarity
                    except KeyError as e:
                        if parent_sense.name() in str(e):
                            break
                        else:
                            continue
            if max_similarity != 0:
                parent_similarities.append(max_similarity)
        if len(parent_similarities) > 0:  # Happens when the cluster has only one item which is not in vocabulary
            parent_similarity = sum(parent_similarities) / len(parent_similarities)
    
    # Similarity between a family and a cluster
    family_similarity = 0
    if not exclude_family:
        family_similarities = []
        for f_item in family:
            for c_item in cluster:
                try:
                    family_similarities.append(own_model.similarity(f_item, c_item))
                except KeyError as e:  # skip the terms not in vocabulary
                    continue
        if len(family_similarities) > 0:
            family_similarity = sum(family_similarities) / len(family_similarities)
    
    # Final score is the average of both the similarities
    return (parent_similarity + family_similarity) / 2


def tune_result(g_improved):
    """ Filter the results i.e. remove all the isolated nodes and nodes with blank labels """

    print('\nTuning the result...')

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


def get_line_count(file_name):
    """ Counts the number of lines in a file using bash """

    return int(subprocess.check_output(
        "wc -l {file_name} | grep -o -E '^[0-9]+'".format(file_name=file_name), shell=True
    ).decode('utf-8').split('\n')[0])


def calculate_f1_score(system_generated_taxo):
    """ Calculate the F1 score of the re-generated taxonomies """

    eval_tool = 'eval/taxi_eval_archive/TExEval.jar'
    eval_gold_standard = 'eval/taxi_eval_archive/input/gold.taxo'
    eval_root = 'science'
    eval_jvm = '-Xmx9000m'
    eval_tool_result = 'out/' + system_generated_taxo.split('/')[-1] + '.evalresult.txt'

    # Running the tool
    tool_command = """java {eval_jvm} -jar {eval_tool} {system_generated_taxo} {eval_gold_standard} {eval_root} {eval_tool_result}""".format(
        eval_jvm=eval_jvm,
        eval_tool=eval_tool,
        system_generated_taxo=system_generated_taxo,
        eval_gold_standard=eval_gold_standard,
        eval_root=eval_root,
        eval_tool_result=eval_tool_result
    )
    print('\nRunning eval-tool:', tool_command)
    subprocess.check_output(tool_command, shell=True)
    print('\nResult of eval-tool written to:', eval_tool_result)

    # Calculating Precision, F1 score and F&M Measure
    l_gold = get_line_count(eval_gold_standard)
    l_input = get_line_count(system_generated_taxo)

    recall = float(subprocess.check_output(
        "tail -n 1 {eval_tool_result} | grep -o -E '[0-9]+[\.]?[0-9]*'".format(
            eval_tool_result=eval_tool_result
        ), shell=True
    ).decode('utf-8').split('\n')[0])
    precision = recall * l_gold / l_input

    f1 = 2 * recall * precision / (recall + precision)
    f_m = float(subprocess.check_output(
        "cat {eval_tool_result} | grep -o -E 'Cumulative Measure.*' | grep -o -E '0\.[0-9]+'".format(
            eval_tool_result=eval_tool_result
        ), shell=True
    ).decode('utf-8').split('\n')[0])

    # Display results
    print('\nRecall:', recall)
    print('Precision:', precision)
    print('F1:', f1)
    print('F&M:', f_m)


def apply_distributional_semantics(nx_graph, taxonomy, mode, depth, iterations, buffer, exclude_parent, exclude_family):
    # Load the pre-trained vectors
    print('Loading embeddings...')
    poincare_w2v, own_w2v = load_vectors()
    print('Loaded.')

    print('\n\nApplying distributional semantics...')
    g_improved = nx_graph.copy()
    clusters_touched = []
    for i in range(1, iterations + 1):
        print('\n\nIteration %d/%d:' % (i, iterations))

        # Remove small clusters
        g_improved, removed_clusters = remove_clusters(own_w2v, g_improved, clusters_touched, depth, buffer)
        print('\nRemoved %d clusters.' % (len(removed_clusters)))
        print('Clusters Removed:', removed_clusters)
        if len(removed_clusters) == 0:
            print('No more clusters left to remove')
            break
        clusters_touched.extend(removed_clusters)  # To ensure that the same cluster does not get removed again

        # Reattach the removed clusters
        if mode == 'reattach':
            print('\nReattaching removed clusters...')
            g_detached = create_children_clusters(own_w2v, g_improved, depth)
            for cluster in removed_clusters:
                max_score = 0
                max_score_node = ''
                for node, graph in g_detached.items():
                    gc = chinese_whispers(graph, weighting='top', iterations=60)
                    for _, family in aggregate_clusters(gc).items():
                        score = calculate_similarity(poincare_w2v, own_w2v, node, family, cluster, exclude_parent, exclude_family)
                        if score > max_score:
                            max_score = score
                            max_score_node = node
                for item in cluster:
                    g_improved.add_edge(max_score_node, item)
        print('Done.')

        # Tune the result
        g_improved = tune_result(g_improved)
        print('Tuned.')

        # Save the results after each iteration and display the F1 score
        save_result(g_improved, taxonomy, mode)


def save_result(result, path, mode):
    print('\nSaving the result...')
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

    # Display the F1 score for the re-generated taxonomy
    calculate_f1_score(output_path)


def main(taxonomy, mode, depth, iterations, buffer, exclude_parent, exclude_family):

    # Read the input
    graph = process_input(taxonomy)

    # Distributional Semantics
    apply_distributional_semantics(graph, taxonomy, mode, depth, iterations, buffer, exclude_parent, exclude_family)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributional Semantics for Taxonomy')
    parser.add_argument('-t', '--taxonomy', required=True, help='Input file containing the taxonomy')
    parser.add_argument(
        '-m', '--mode', default='reattach', choices=['only_removal', 'reattach'],
        help='Mode of execution: Only remove the nodes or reattach the removed nodes.'
    )
    parser.add_argument(
        '-d', '--depth', type=int, default=100,
        help='Number of results to return while checking for most similar nodes of a term.'
    )
    parser.add_argument('-i', '--iterations', type=int, default=1, help='Number of iterations.')
    parser.add_argument('-b', '--buffer', type=int, default=10, help='Number of clusters to remove per iteration')
    parser.add_argument('-p', '--parent', action='store_false', help='Exculde "parent" while calculating cluster similarity')
    parser.add_argument('-f', '--family', action='store_false', help='Exclude "family" while calculating cluster similarity')
    args = parser.parse_args()

    if not args.parent and not args.family:
        parser.error("""Both --parent(-p) and --family(-f) cannot be set to False.
        Run: 'python distributional_semantics.py --help' for more options.""")

    print('Input File:', args.taxonomy)
    print('Mode:', args.mode)

    main(args.taxonomy, args.mode, args.depth, args.iterations, args.buffer, args.parent, args.family)
