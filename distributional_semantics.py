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


def create_children_clusters(own_model, graph):
    """ This function returns a dictionary where corresponding to each key(node) is a graph of its children """
    clustered_graph = {}
    for node in graph.nodes():
        clustered_graph[node] = nx.Graph()
        successors = [s.lower() for s in graph.successors(node)]

        for successor in successors:
            try:
                for word, score in own_model.most_similar(successor, topn=100):
                    if word.lower() in successors:
                        clustered_graph[node].add_edge(successor, word.lower())
            except KeyError:  # If the word in not in vocabulary, check using the substring based method
                successor_terms = successor.split('_')
                root_terms = [successor_terms[0], successor_terms[-1]]
                if node in root_terms:
                    clustered_graph[node].add_node(successor)
    
    return clustered_graph


def remove_clusters(own_model, nx_graph, clusters_touched, buffer):
    """ Removes the less related and small clusters from the graph """

    print('Removing small clusters..')
    g_clustered = create_children_clusters(own_model, nx_graph)
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


def graph_pruning(path, output_dir, domain):

    cycle_removing_tool = "graph_pruning/graph_pruning.py"
    cycle_removing_method = "tarjan"
    cleaning_tool = "graph_pruning/cleaning.py"

    input_file = os.path.basename(path)
    file_pruned_out = input_file + '-pruned.csv'
    file_cleaned_out = file_pruned_out + '-cleaned.csv'

    print('======================================================================================================================')
    cycle_removing_cmd = 'python {cycle_removing_tool} {input_path} {output_dir}/{file_pruned_out} {cycle_removing_method} | tee /dev/tty'.format(
        cycle_removing_tool=cycle_removing_tool,
        input_path=path,
        output_dir=output_dir,
        file_pruned_out=file_pruned_out,
        cycle_removing_method=cycle_removing_method
    )
    print('Cycle removing:', cycle_removing_cmd, sep='\n')
    subprocess.check_output(cycle_removing_cmd, shell=True)
    print('Cycle removing finished. Written to: {output_dir}/{file_pruned_out}\n'.format(
        output_dir=output_dir, file_pruned_out=file_pruned_out
    ))

    print('======================================================================================================================')
    cleaning_cmd = 'python {cleaning_tool} {output_dir}/{file_pruned_out} {output_dir}/{file_cleaned_out} {domain}'.format(
        cleaning_tool=cleaning_tool,
        output_dir=output_dir,
        file_pruned_out=file_pruned_out,
        file_cleaned_out=file_cleaned_out,
        domain=domain
    )
    print('Cleaning:', cleaning_cmd, sep='\n')
    subprocess.check_output(cleaning_cmd, shell=True)
    print('Finished cleaning. Write output to: {output_dir}/{file_cleaned_out}\n'.format(
        output_dir=output_dir, file_cleaned_out=file_cleaned_out
    ))

    return os.path.join(output_dir, file_cleaned_out)


def calculate_f1_score(system_generated_taxo, output_dir, domain, iteration):
    """ Calculate the F1 score of the re-generated taxonomies """

    eval_tool = 'eval/taxi_eval_archive/TExEval.jar'
    eval_gold_standard = 'eval/taxi_eval_archive/gold_standard/{domain}.taxo'.format(domain=domain)
    eval_root = domain
    eval_jvm = '-Xmx9000m'
    eval_tool_result = os.path.join(output_dir, os.path.basename(system_generated_taxo)) + '-evalresult.txt'

    # Running the tool
    print('======================================================================================================================')
    tool_command = """java {eval_jvm} -jar {eval_tool} {system_generated_taxo} {eval_gold_standard} {eval_root} {eval_tool_result}""".format(
        eval_jvm=eval_jvm,
        eval_tool=eval_tool,
        system_generated_taxo=system_generated_taxo,
        eval_gold_standard=eval_gold_standard,
        eval_root=eval_root,
        eval_tool_result=eval_tool_result
    )
    print('Running eval-tool:', tool_command, sep='\n')
    subprocess.check_output(tool_command, shell=True)
    print('Result of eval-tool written to:', eval_tool_result)

    # Calculating Precision, F1 score and F&M Measure
    scores = {}
    l_gold = get_line_count(eval_gold_standard)
    l_input = get_line_count(system_generated_taxo)

    scores['recall'] = float(subprocess.check_output(
        "tail -n 1 {eval_tool_result} | grep -o -E '[0-9]+[\.]?[0-9]*'".format(
            eval_tool_result=eval_tool_result
        ), shell=True
    ).decode('utf-8').split('\n')[0])
    scores['precision'] = scores['recall'] * l_gold / l_input

    scores['f1'] = 2 * scores['recall'] * scores['precision'] / (scores['recall'] + scores['precision'])
    scores['f_m'] = float(subprocess.check_output(
        "cat {eval_tool_result} | grep -o -E 'Cumulative Measure.*' | grep -o -E '0\.[0-9]+'".format(
            eval_tool_result=eval_tool_result
        ), shell=True
    ).decode('utf-8').split('\n')[0])

    # Display results
    print('\nPrecision:', scores['precision'])
    print('Recall:', scores['recall'])
    print('F1:', scores['f1'])
    print('F&M:', scores['f_m'])

    return scores


def apply_distributional_semantics(nx_graph, taxonomy, mode, domain, iterations, buffer, exclude_parent, exclude_family):
    # Load the pre-trained vectors
    print('Loading embeddings...')
    poincare_w2v, own_w2v = load_vectors()
    print('Loaded.')

    print('\n\nApplying distributional semantics...')
    scores = {}
    output_dir = 'out'
    g_improved = nx_graph.copy()
    clusters_touched = []
    for i in range(1, iterations + 1):
        print('\n\nIteration %d/%d:' % (i, iterations))

        # Remove small clusters
        g_improved, removed_clusters = remove_clusters(own_w2v, g_improved, clusters_touched, buffer)
        print('\nRemoved %d clusters.' % (len(removed_clusters)))
        print('Clusters Removed:', removed_clusters)
        if len(removed_clusters) == 0:
            print('No more clusters left to remove')
            break
        clusters_touched.extend(removed_clusters)  # To ensure that the same cluster does not get removed again

        # Reattach the removed clusters
        if mode == 'reattach':
            print('\nReattaching removed clusters...')
            g_detached = create_children_clusters(own_w2v, g_improved)
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
        output_path = save_result(g_improved, taxonomy, mode)

        # Prune and clean the generated taxonomy
        pruned_output = graph_pruning(output_path, output_dir, domain)

        # Display the F1 score for the generated taxonomy
        scores[i] = calculate_f1_score(pruned_output, output_dir, domain, i)
    
    # Write the scores of each iteration in a CSV file
    with open(os.path.join(output_dir, os.path.basename(taxonomy)) + '-iter-records.csv', 'w') as f:
        f.write('iteration,precision,recall,f1,f_m\n')
        for iter in scores:
            f.write(
                '{iter},{precision},{recall},{f1},{f_m}\n'.format(
                    iter=iter,
                    precision=scores[iter]['precision'],
                    recall=scores[iter]['recall'],
                    f1=scores[iter]['f1'],
                    f_m=scores[iter]['f_m']
                )
            )


def save_result(result, path, mode):
    print('\nSaving the result...')
    df_improved = pd.DataFrame(list(result.edges()), columns=['hypernym', 'hyponym'])
    df_improved = df_improved[df_improved.columns.tolist()[::-1]]

    # Replace the underscores with blanks
    df_improved['hyponym'] = df_improved['hyponym'].apply(lambda x: x.replace('_', ' '))
    df_improved['hypernym'] = df_improved['hypernym'].apply(lambda x: x.replace('_', ' '))

    # Store the result
    output_path = os.path.join(
        'taxi_output', 'distributional_semantics',
        os.path.basename(path) + '-' + mode + os.path.splitext(path)[-1]
    )
    df_improved.to_csv(output_path, sep='\t', header=False)
    print('Output saved at:', output_path)

    return output_path


def main(taxonomy, mode, domain, iterations, buffer, exclude_parent, exclude_family):

    # Read the input
    graph = process_input(taxonomy)

    # Distributional Semantics
    apply_distributional_semantics(graph, taxonomy, mode, domain, iterations, buffer, exclude_parent, exclude_family)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributional Semantics for Taxonomy')
    parser.add_argument('-t', '--taxonomy', required=True, help='Input file containing the taxonomy')
    parser.add_argument(
        '-m', '--mode', default='reattach', choices=['only_removal', 'reattach'],
        help='Mode of execution: Only remove the nodes or reattach the removed nodes.'
    )
    parser.add_argument(
        '-d', '--domain', default='science',
        choices=['science', 'science_wordnet', 'science_eurovoc', 'food', 'food_wordnet', 'environment_eurovoc'],
        help='Domain of the taxonomy'
    )
    parser.add_argument('-i', '--iterations', type=int, default=1, help='Number of iterations.')
    parser.add_argument('-b', '--buffer', type=int, default=10, help='Number of clusters to remove per iteration')
    parser.add_argument('-p', '--parent', action='store_false', help='Exculde "parent" while calculating cluster similarity')
    parser.add_argument('-f', '--family', action='store_false', help='Exclude "family" while calculating cluster similarity')
    args = parser.parse_args()

    if not args.parent and not args.family:
        parser.error("""Both --parent(-p) and --family(-f) cannot be set to False.
        Run: 'python distributional_semantics.py --help' for more options.""")
    
    print('Domain:', args.domain)
    print('Input File:', args.taxonomy)
    print('Mode:', args.mode)

    main(args.taxonomy, args.mode, args.domain, args.iterations, args.buffer, args.parent, args.family)
