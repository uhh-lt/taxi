import networkx as nx
import methods.util.write_graph as write_graph
import methods.util.util as util
from zhenv5.remove_cycle_edges_by_hierarchy_greedy import scc_based_to_remove_cycle_edges_iterately
from zhenv5.remove_cycle_edges_by_hierarchy_BF import remove_cycle_edges_BF_iterately
from zhenv5.remove_cycle_edges_by_hierarchy_voting import remove_cycle_edges_heuristic

SUPPORTED_RANKING_METHODS = ["greedy", "forward", "backward", "voting"]

g = nx.DiGraph()


def prepare(line):
    g.add_edge(line[1], line[2])


def do(filename_out, delimiter, mode, gephi_out):
    inputs = mode.split("_")

    if len(inputs) < 2:
        raise Exception(
            "No score method provided (e.g. 'hierarchy_pagerank_voting'). Supported: pagerank, socialagony, trueskill")

    if len(inputs) < 3 or inputs[2] not in SUPPORTED_RANKING_METHODS:
        raise Exception("Ranking method '%s' not supported. Supported: %s" % (inputs[2], SUPPORTED_RANKING_METHODS))

    score_name = inputs[1]
    ranking = inputs[2]

    print("Score method: %s" % score_name)
    print("Ranking method: %s" % ranking)

    players_score_dict = computing_hierarchy(filename_out, score_name)
    edges_to_remove, e1, e2, e3, e4 = None, None, None, None, None

    if ranking == "voting" or ranking == "greedy":
        e1 = scc_based_to_remove_cycle_edges_iterately(g.copy(), players_score_dict)
        edges_to_remove = e1

    if ranking == "voting" or ranking == "forward":
        e2 = remove_cycle_edges_BF_iterately(g.copy(), players_score_dict, is_Forward=True, score_name=score_name)
        edges_to_remove = e2

    if ranking == "voting" or ranking == "backward":
        e3 = remove_cycle_edges_BF_iterately(g.copy(), players_score_dict, is_Forward=False, score_name=score_name)
        edges_to_remove = e3

    if ranking == "voting":
        e4 = remove_cycle_edges_by_voting([set(e1), set(e2), set(e3)])
        edges_to_remove = e4

    # e1, e2, e3, e4 = remove_cycle_edges_by_hierarchy(filename_out, players_score_dict, players_score_name)

    util.remove_edges_from_network_graph(g, edges_to_remove)
    write_graph.network_graph(filename_out, g, gephi_out=gephi_out, delimiter=delimiter)



def dir_tail_name(file_name):
    import os.path
    dir_name = os.path.dirname(file_name)
    head, tail = os.path.split(file_name)
    print("dir name: %s, file_name: %s" % (dir_name, tail))
    return dir_name, tail


def get_edges_voting_scores(set_edges_list):
    total_edges = set()
    for edges in set_edges_list:
        total_edges = total_edges | edges
    edges_score = {}
    for e in total_edges:
        edges_score[e] = len(filter(lambda x: e in x, set_edges_list))
    return edges_score


def remove_cycle_edges_strategies(graph_file, nodes_score_dict, score_name="socialagony", nodetype=int):
    # greedy
    cg = g.copy()
    e1 = scc_based_to_remove_cycle_edges_iterately(cg, nodes_score_dict)

    # forward
    cg = g.copy()
    e2 = remove_cycle_edges_BF_iterately(cg, nodes_score_dict, is_Forward=True, score_name=score_name)

    # backward
    cg = g.copy()
    e3 = remove_cycle_edges_BF_iterately(cg, nodes_score_dict, is_Forward=False, score_name=score_name)

    return e1, e2, e3


def remove_cycle_edges_by_voting(set_edges_list, nodetype=int):
    edges_score = get_edges_voting_scores(set_edges_list)
    e = remove_cycle_edges_heuristic(g.copy(), edges_score, nodetype=nodetype)
    return e


def remove_cycle_edges_by_hierarchy(graph_file, nodes_score_dict, score_name="socialagony"):
    e1, e2, e3 = remove_cycle_edges_strategies(graph_file, nodes_score_dict, score_name=score_name)
    e4 = remove_cycle_edges_by_voting([set(e1), set(e2), set(e3)])
    return e1, e2, e3, e4


def computing_hierarchy(graph_file, players_score_func_name, nodetype=int):
    import os.path
    if players_score_func_name == "socialagony":
        dir_name, tail = dir_tail_name(graph_file)
        agony_file = os.path.join(dir_name, tail.split(".")[0] + "_socialagony.txt")
        # agony_file = graph_file[:len(graph_file)-6] + "_socialagony.txt"
        # from compute_social_agony import compute_social_agony
        # players = compute_social_agony(graph_file,agony_path = "agony/agony ")
        if False:
            # if os.path.isfile(agony_file):
            print("load pre-computed socialagony from: %s" % agony_file)
            players = read_dict_from_file(agony_file)
        else:
            print("start computing socialagony...")
            # from compute_social_agony import compute_social_agony
            # players = compute_social_agony(graph_file, agony_path="agony/agony ")
            print("write socialagony to file: %s" % agony_file)
        return players
    if players_score_func_name == "pagerank":
        print("computing pagerank...")
        players = nx.pagerank(g, alpha=0.85)
        return players
    elif players_score_func_name == "trueskill":
        output_file = graph_file[:len(graph_file) - 6] + "_trueskill.txt"
        output_file_2 = graph_file[:len(graph_file) - 6] + "_trueskill.pkl"
        # from true_skill import graphbased_trueskill
        # players = graphbased_trueskill(g)
        # from file_io import write_dict_to_file
        # write_dict_to_file(players,output_file)

        '''
        if os.path.isfile(output_file):
            print("load pre-computed trueskill from: %s" % output_file)
            players = read_dict_from_file(output_file,key_type = int, value_type = float)
        elif os.path.isfile(output_file_2):
            print("load pre-computed trueskill from: %s" % output_file_2)
            players = read_from_pickle(output_file_2)			
        '''
        if True:
            print("start computing trueskill...")
            from zhenv5.true_skill import graphbased_trueskill
            players = graphbased_trueskill(g)

        return players
