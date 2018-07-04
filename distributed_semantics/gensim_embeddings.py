from __future__ import print_function
import gensim
import csv
import io
import sys
import numpy as np
import gzip
import os
import argparse
import logging
import pandas

compound_operator = "-"

def create_compound_word(compound, model):
    global compound_operator
    word_parts = compound.split(compound_operator)
    compound_word = np.copy(model.wv[word_parts[0]])
    for i in range(1, len(word_parts)):
        part = word_parts[i]
        compound_word += model.wv[part]
    compound_word /= (len(word_parts))
    return compound_word


def compare_to_gold(gold, taxonomy_o, outliers, model, mode = "removal", log = False, write_file = None):
    taxonomy = taxonomy_o.copy()
    global compound_operator
    removed_outliers = []
    for element in taxonomy:
        if (element[1].replace(' ', compound_operator), element[2].replace(' ', compound_operator)) in outliers:
            #print("skip: " + element[1] + " " + element[2])
            if mode == "removal_add":
                best_word, parent, rank, rank_inv, rank_root = connect_to_taxonomy(taxonomy.copy(),element[1].replace(' ', compound_operator), model)
                if rank != None and rank_inv != None:
                    rank_ref =  rank + rank_inv
                    if  not rank_ref > 150:
                        if rank_root != None and rank_root in range(rank_ref -20, rank_ref + 20):
                        #and rank_root in range(rank_ref -20, rank_ref + 20):
                            removed_outliers.append((element[0], element[1], parent.replace(compound_operator, ' ')))
                            print("Added :" + str(element[0]) + " " + element[1].replace(compound_operator, ' ') + " " +  parent.replace(compound_operator, ' '))
                            print("Best Word: " + best_word + ", Rank:" + str(rank) + " Rank_Inv: " + str(rank_inv) + ", Rank Parent: " + str(rank_root))

                # elif rank_root == "None":
                #     removed_outliers.append(element)
            continue
        removed_outliers.append(element)

    correct = 0
    for element in removed_outliers:
        for ele_g in gold:
            if element[1] == ele_g[1] and element[2] == ele_g[2]:
                correct+=1
                break
    precision = correct / float(len(removed_outliers))
    recall = correct / float(len(gold))
    print(float(len(removed_outliers)))
    print(float(len(gold)))
    print("Correct: " + str(correct))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(2*precision *recall / (precision + recall)))
    if log != None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log)
        with open(path + ".txt", 'w') as f:
            for element in outliers:
                f.write(element[0] + '\t' + element[1] + '\n')
            f.write("Elements Taxonomy:" + str(float(len(removed_outliers))))
            f.write(str((float(len(gold)))) + '\n')
            f.write("Correct: " + str(correct) + '\n')
            f.write("Precision: " + str(precision) + '\n')
            f.write("Recall: " + str(recall) + '\n')
            f.write("F1: " + str(2*precision *recall / (precision + recall)) + '\n')
            f.close()
    if write_file != None:
        path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), write_file + ".csv")
        with open(path, 'w') as f:
            for element in removed_outliers:
                f.write(element[0] + '\t' + element[1] + '\t' + element[2]  + '\n')
        f.close()

    return removed_outliers




def read_trial_data():
    all_info = []
    trial_dataset_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"relations.csv")
    with open(trial_dataset_fpath, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            #id, hyponym, hypernym, correct, source
            all_info.append((line[0], line[1], line[2], line[3], line[4]))
    correct_relations = []
    wrong_relations = []
    all_relations = []
    taxonomy = []
    for entry in all_info:
        if entry[4] in ["WN_plants.taxo", "WN_vehicles.taxo", "ontolearn_AI.taxo"]: #alternatively add negative co-hypo
            if entry[3] == "1":
                correct_relations.append((entry[0], entry[1], entry[2]))
                all_relations.append((entry[0], entry[1], entry[2]))
            else:
                wrong_relations.append((entry[0], entry[1], entry[2]))
                all_relations.append((entry[0], entry[1], entry[2]))

    for i in range(len(all_relations)):
        all_relations[i] = (all_relations[i][0], all_relations[i][1].replace(" ", compound_operator), all_relations[i][2].replace(" ", compound_operator))
    return [correct_relations, all_relations, taxonomy]

def read_all_data():
    global compound_operator
    # Load Google's pre-trained Word2Vec model.
    filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../out/science_en.csv-relations.csv-taxo-knn1.csv-pruned.csv-cleaned.csv")
    filename_gold = "gold.taxo"
    # Load Google's pre-trained Word2Vec model.
    relations = []
    with open(filename_in, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            relations.append((line[0], line[1], line[2]))

    gold= []
    with open(filename_gold, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            gold.append((line[0], line[1], line[2]))
    return [gold, relations]

def get_parent(relations,child):
    for relation in relations:
        if child == relation[1]:
            return relation[2]
    return None

def get_rank(entity1, entity2, model, threshhold):
    rank_inv = None
    similarities_rev = model.wv.similar_by_word(entity1, threshhold)
    similarities_rev = [entry[0] for entry in similarities_rev]
    for j in range(len(similarities_rev)):
        temp_rev = similarities_rev[j]
        if entity2 == temp_rev:
            rank_inv = j
        # selectors  = ["computer science", "computer-science", "computer_science"]
        # for selector in selectors:
        #     if entity2 == selector:
        #         print(selector)
    return rank_inv

def connect_to_taxonomy(relations_o, current_word, model):
    relations = relations_o.copy()
    global compound_operator
    for i in range(len(relations)):
        relations[i] = (relations[i][0], relations[i][1].replace(" ", compound_operator), relations[i][2].replace(" ", compound_operator))
    words_o = [relation[2] for relation in relations] + [relation[1] for relation in relations]
    words_a = [relation[2] for relation in relations if relation[2] in model.wv] + [relation[1] for relation in relations if relation[1] in model.wv]
    #print("Original" + str(len(words_o)) + "Remaining: " + str(len(words_a)))
    words_a = list(set(words_a))
    best_word = None
    #print(current_word)
    if not current_word in model.wv:
        print("outlier word not found in voc")
        return
    words_a.remove(current_word)
    element = model.wv.most_similar_to_given(current_word, words_a)
    while get_parent(relations, element) == current_word:
        words_a.remove(element)
        element = model.wv.most_similar_to_given(current_word, words_a)
    #print(current_word + " " + element)
    #curr_rank = model.wv.closer_than(current_word, element)
    rank = get_rank(current_word, element, model, 100000)
    if rank != None:
        best_word = element
    rank_inv = get_rank(element, current_word, model, 100000)
    parent =  get_parent(relations, element)
    rank_root = get_rank(current_word, parent , model, 100000)

    #print("Rank :" + str(rank) + ", Rank_Iverse:"+ str(rank_inv) +  ", Rank root: " + str(rank_root) + ", highest similarity: " + best_word + " " + current_word + ", parent: " +  parent)
    return [best_word, parent, rank, rank_inv, rank_root]


#since titles are just mashed into string consider in the future to find a way to detect the title
def remove_title(text):
    curr_title = ""
    seperated_text = text.split(" ")
    for i in range(len(seperated_text)):
        word = seperated_text[i]
        if word == curr_title.split(" ")[0]:
            break
        else:
            curr_title = curr_title + word[i]
    return text.remove(curr_title)


def read_input(input_file, vocabulary):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    colnames = ["id,", "text"]
    data = pandas.read_csv(input_file, names= colnames)
    text = data.text.tolist()
    print("Number of Reviews: " + str(len(text)))
    for i in range(len(text)):
        line = text[i]
        if (i%10000==0):
            logging.info ("read {0} reviews".format (i))
            print(line)
        # do some pre-processing and return a list of words for each review text
        line = line.lower()
        for word_voc in vocabulary:
            if word_voc in line:
                if i == 40038:
                    print(word_voc + " " + str(i))
                    line = line.replace(word_voc, word_voc.replace(' ', compound_operator))
                    print(line)

        cleared_line = gensim.utils.simple_preprocess (line)
        yield cleared_line


def visualize_taxonomy(taxonomy_vectors, taxonomy_names):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(taxonomy_vectors)

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(taxonomy_names, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()




def calculate_outliers(relations_o, model, mode, embedding_type = None, threshhold = None):
    relations = relations_o.copy()
    structure = {}
    outliers = []
    for i in range(len(relations)):
        relations[i] = (relations[i][0], relations[i][1].replace(" ", compound_operator), relations[i][2].replace(" ", compound_operator))

    for parent in [relation[2] for relation in relations]:
        structure[parent] = [relation[1] for relation in relations if relation[2] == parent]

    for key in structure:
        data_base_word_name = key
        if structure[key] == []:
            print("no children: " + key)
            continue
        # if not key in model.wv and key.title() in model.wv:
        #     print "Uppercase in Model: " + key.title()
        #     data_base_word_name = key.title()
        elif not key in model.wv:
            continue
        cleaned_co_hyponyms = []
        for word in structure[key]:
            if word in model.wv:
                cleaned_co_hyponyms.append(word)
        if len(cleaned_co_hyponyms) < 1:
            continue
        #print(cleaned_co_hyponyms)
        above_treshhold = False
        while not above_treshhold:
            outlier = model.wv.doesnt_match(cleaned_co_hyponyms)
            sim = model.wv.similarity(data_base_word_name, outlier)
            #print(key + " " + outlier)
            if threshhold == None:
                if embedding_type == "0":
                    threshhold = 0.35#0.51
                elif embedding_type == "1":
                    threshhold = 0.6 #0.20
                else:
                    threshhold = 0.5

            #
            threshhold_bool = False
            if mode == "k_nearest":
                threshhold_bool = model.wv.rank(key, outlier) > threshhold and model.wv.rank(outlier, key) > threshhold
            if mode == "abs":
                threshhold_bool = sim < threshhold
            if threshhold_bool:
                outliers.append((outlier, key))
                cleaned_co_hyponyms.remove(outlier)
                if len(cleaned_co_hyponyms) < 1:
                    #print(str(sim) + '\n')
                    break
            else:
                above_treshhold = True
            #print(str(sim) + '\n')
    print(outliers)
    return outliers

embedding = None


if len(sys.argv) >= 2:
    mode = sys.argv[1]

if len(sys.argv) >= 3:
    embedding = sys.argv[2]




def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('mode', type=str, default='preload', choices=["normal", "train_embeddings", "gridsearch_removal", "gridsearch_removal_add", "gridsearch_removal_add_iterative"], help="Mode of the system.")
    parser.add_argument('embedding', type=str, default='quick', choices=["fasttext_ref", "wiki2M", "wiki1M_subword", "own_fasttext","own_w2v", "quick"], help="Classifier architecture of the system.")
    parser.add_argument('experiment_name', type=str, default=None, help="Name of the Experiment")
    parser.add_argument('--log', action='store_true', help="Logs taxonomy and results")
    parser.add_argument('--trial', action='store_true', help="Uses trial dataset")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.embedding, args.experiment_name, args.log, args.trial)


def run(mode, embedding, experiment_name, log = False, trial = False):
    if embedding == "fasttext_ref":
        #model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', binary=False)
        model = gensim.models.FastText.load_fasttext_format('wiki.en.bin')
        #model = gensim.models.FastText.load_fasttext_format('crawl-300d-2M.vec')
    elif embedding == "wiki2M":
        #model = gensim.models.FastText.load_fasttext_format('crawl-300d-2M.vec','vec')
        model = gensim.models.KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False)
        #model.save("crawl-300d-2M.bin")
    elif embedding == "wiki1M_subword":
        model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', binary=False)

    elif embedding == "own_fasttext":
        model = gensim.models.FastText.load("own_embeddings")
        print(model.wv.similarity("dog", "cat"))
        print(model.wv.similarity("clean", "dirty"))
        print(model.wv.similarity("computer-science", "science"))
        print(model.wv.similarity("signal-processing", "electrical-engineering"))
        print(model.wv.similarity("plant-breeding", "plant-science"))
        print(model.wv.similarity('digital-circuits', 'electrical-engineering'))
        print(len(model.wv.vocab))
    elif embedding == "own_w2v":
        model = gensim.models.KeyedVectors.load('own_embeddings_w2v')
        print(model.wv.similarity("dog", "cat"))
        print(model.wv.similarity("clean", "dirty"))
        print(model.wv.similarity("signal-processing", "dog"))
        #print(model.wv.similarity("computer-science", "science"))
        #print(model.wv.similarity("signal-processing", "electrical-engineering"))
        #print(model.wv.similarity("plant-breeding", "plant-science"))
        #print(model.wv.similarity('digital-circuits', 'electrical-engineering'))
        print(len(model.wv.vocab))
    elif embedding == "quick":
        model = gensim.models.KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False, limit = 50000)

    gold = []
    relations = []
    taxonomy = []
    outliers = []
    if mode =="train_embeddings":
        gold,relations = read_all_data()
        vocabulary = set([relation[2] for relation in relations] + [relation[1] for relation in relations])
        documents = list(read_input(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikipedia_utf8_filtered_20pageviews.csv" ),vocabulary))
        #documents = list(read_input(train_data_raw,vocabulary))
        model = gensim.models.Word2Vec(size= 300, window = 5, min_count = 5, workers = 30)
        model.build_vocab(documents)
        #model.train(documents, total_examples = len(documents), epochs=10)
        model.train(documents, total_examples=model.corpus_count, epochs=6)
        model.save("own_embeddings_w2v")

    if not trial:
        if mode == "normal":
            gold, relations = read_all_data()
            for i in range(1,10):
                print(len(relations))
                outliers = calculate_outliers(relations, model, mode = "abs", embedding_type = embedding)
                relations = compare_to_gold(gold, relations,  outliers, model, write_file = "out/test")



        elif mode =="gridsearch_removal":
            threshholds = range(2,8, 1)
            threshholds = [float(value / 10) for value in threshholds]
            #threshholds = [0.6, 0.62, 0.63, 0.7, 0.95]
            for value in threshholds:
                gold, relations = read_all_data()
                outliers = calculate_outliers(relations,model, mode = "abs", embedding_type = embedding, threshhold=  value)
                compare_to_gold(gold, relations, outliers, model, mode = "removal", log  = "logs/" + experiment_name + "_" + str(value), write_file = "out/" + experiment_name + "_" + str(value))

        elif mode =="gridsearch_removal_add":
            threshholds = range(2,8)
            threshholds = [float(value / 10) for value in threshholds]
            #threshholds = [0.7, 0.74, 0.75, 0.8]
            for value in threshholds:
                gold, relations = read_all_data()
                outliers = calculate_outliers(relations,model, mode = "abs", embedding_type = embedding, threshhold=  value)
                compare_to_gold(gold, relations, outliers, model, mode = "removal_add", log  = "logs/" + experiment_name + "_" + str(value), write_file = "out/" + experiment_name + "_" + str(value))



        elif mode =="gridsearch_removal_add_iterative":
            threshholds = range(100, 1000, 100)
            #threshholds = [float(value / 10) for value in threshholds]
            for value in threshholds:
                gold, relations  = read_all_data()
                for i in range(1,3):
                    outliers = calculate_outliers(relations,model, "k_nearest", embedding_type = embedding , threshhold = value)
                    relations = compare_to_gold(gold, relations, outliers, model, mode = "removal_add", log = True, experiment_name = "logs/wikipedia_2M_outlier_removal_by_rank_adding_back__by_rank_iterative_3/", threshhold = value)


if __name__ == '__main__':
    main()


# def valid_words(relations, model):
#     global compound_operator
#     valid_words = set([])
#     compound_words = {}
#     for relation in relations:
#         issubset_1 = True
#         issubset_2 = True
#         for entry in relation[1].split(compound_operator):
#             if not entry in model.wv:
#                 issubset_1 = False
#                 break
#         if relation[1] in model.wv:
#             valid_words.add(relation[1])
#         elif issubset_1:
#             #print word
#             compound_word = create_compound_word(relation[1], model)
#             compound_words[relation[1]] = compound_word
#             valid_words.add(relation[1])
#
#         for entry in relation[2].split(compound_operator):
#             if not entry in model.wv:
#                 issubset_2 = False
#                 break
#
#         if relation[2] in model.wv:
#             valid_words.add(relation[2])
#         elif issubset_2:
#             #print word
#             compound_word = create_compound_word(relation[2], model)
#             compound_words[relation[2]] = compound_word
#             valid_words.add(relation[2])
#             #model.syn0.build_vocab([relation[2]], update=True)
#             model.syn0[relation[2]] = compound_word
#
#     return valid_words, compound_words
