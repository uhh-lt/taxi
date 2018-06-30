from __future__ import print_function
import gensim
import csv
import io
import sys
import numpy as np
import gzip
import os
import logging

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


def compare_to_gold(gold, taxonomy, outliers, model, log = False, experiment_name = "not_specified", threshhold = None):
    global compound_operator
    removed_outliers = []
    for element in taxonomy:
        if (element[1], element[2]) in outliers:
            #print("skip: " + element[1] + " " + element[2])
            best_word, parent, rank, rank_inv, rank_root = connect_to_taxonomy(taxonomy.copy(),element[1].replace(' ', compound_operator), model)
            if not rank > 100 and not rank_inv > 100:
                if rank_root != "None" and rank_root < 100:
                    removed_outliers.append((element[0], element[1].replace(compound_operator, ' '), parent.replace(compound_operator, ' ')))
                    print("Added :" + str(element[0]) + " " + element[1].replace(compound_operator, ' ') + " " +  parent.replace(compound_operator, ' '))
                    print("Best Word: " + best_word + ", Rank:" + str(rank) + " Rank_Inv: " + str(rank_inv) + ", Rank Parent: " + str(rank_root))

                # elif rank_root == "None":
                #     removed_outliers.append(element)
            continue
        removed_outliers.append((element[0], element[1].replace(compound_operator, ' '), element[2].replace(compound_operator, ' ')))

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
    if log:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), experiment_name)
        if not os.path.exists(path):
                os.makedirs(path)
        with open(os.path.join(path, str(threshhold) + ".txt"), 'w') as f:
            for element in outliers:
                f.write(element[0] + '\t' + element[1] + '\n')
            f.write("Elements Taxonomy:" + str(float(len(removed_outliers))))
            f.write(str((float(len(gold)))) + '\n')
            f.write("Correct: " + str(correct) + '\n')
            f.write("Precision: " + str(precision) + '\n')
            f.write("Recall: " + str(recall) + '\n')
            f.write("F1: " + str(2*precision *recall / (precision + recall)) + '\n')
            f.close()

    for i in range(len(removed_outliers)):
        removed_outliers[i] = (removed_outliers[i][0], removed_outliers[i][1].replace(" ", compound_operator), removed_outliers[i][2].replace(" ", compound_operator))
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
                taxonomy.append((entry[0], entry[1], entry[2]))
            else:
                wrong_relations.append((entry[0], entry[1], entry[2]))
                all_relations.append((entry[0], entry[1], entry[2]))
                taxonomy.append((entry[0], entry[1], entry[2]))

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
    taxonomy = []
    with open(filename_in, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            relations.append((line[0], line[1], line[2]))
            taxonomy.append((line[0], line[1], line[2]))

    for i in range(len(relations)):
        relations[i] = (relations[i][0], relations[i][1].replace(" ", compound_operator), relations[i][2].replace(" ", compound_operator))


    gold= []
    with open(filename_gold, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            gold.append((line[0], line[1], line[2]))
    return [gold, relations, taxonomy]

def get_parent(relations,child):
    for relation in relations:
        if child == relation[1]:
            return relation[2]
    return None

def get_rank(entity1, entity2, model, threshhold):
    rank_inv = "None"
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

def connect_to_taxonomy(relations, current_word, model):
    global compound_operator
    for i in range(len(relations)):
        relations[i] = (relations[i][0], relations[i][1].replace(" ", compound_operator), relations[i][2].replace(" ", compound_operator))
    words_o = [relation[2] for relation in relations] + [relation[1] for relation in relations]
    words_a = [relation[2] for relation in relations if relation[2] in model.wv] + [relation[1] for relation in relations if relation[1] in model.wv]
    #print("Original" + str(len(words_o)) + "Remaining: " + str(len(words_a)))
    words_a = list(set(words_a))
    best_word = "None"
    #print(current_word)
    if not current_word in model.wv:
        print("outlier word not found in voc")
        return
    words_a.remove(current_word)
    element = model.wv.most_similar_to_given(current_word, words_a)
    #print(current_word + " " + element)
    #curr_rank = model.wv.closer_than(current_word, element)
    rank = get_rank(current_word, element, model, 100000)
    if rank != "None":
        best_word = element
    rank_inv = get_rank(element, current_word, model, 100000)
    parent =  get_parent(relations, element)
    rank_root = get_rank(current_word, parent , model, 100000)

    #print("Rank :" + str(rank) + ", Rank_Iverse:"+ str(rank_inv) +  ", Rank root: " + str(rank_root) + ", highest similarity: " + best_word + " " + current_word + ", parent: " +  parent)
    return [best_word, parent, rank, rank_inv, rank_root]

    # if element in similarities :
    #     best_word = element
    # for word_i in words_a:
    #     print(word_i)
    #     if word_i in model.wv and word_i != current_word:
    #         #curr_rank = model.wv.rank(word_i, current_word)
    #         if curr_rank < best_rank:
    #             best_rank = curr_rank
    #             best_word = word_i
    #     else:
    #         continue

def read_input(input_file, vocabulary):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))

    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate (f):
            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            line = line.lower()
            # for word_voc in vocabulary:
            #     line.replace(str.encode(word_voc), str.encode(word_voc.replace(' ', compound_operator)))
            cleared_line = gensim.utils.simple_preprocess (line)
            yield cleared_line



# def create_data_set_embedding(data_path):
#     data_file = data_path
#     with gzip.open ('reviews_data.txt.gz', 'rb') as f:
#     for i,line in enumerate (f):
#         print(line)
#         break
#
# def train_embedding:




def calculate_outliers(relations, model, mode, threshhold = None):
    structure = {}
    outliers = []
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
                if mode == "0":
                    threshhold = 0.35#0.51
                elif mode == "1":
                    threshhold = 0.6 #0.20
                else:
                    threshhold = 0.5

            #if sim < threshhold:
            if model.wv.rank(key, outlier) > threshhold:
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

mode = None

if len(sys.argv) >= 2:
    mode = sys.argv[1]

if len(sys.argv) >= 3:
    embedding = sys.argv[2]

gold = []
relations = []
taxonomy = []
model = None
if embedding == "0":
    #model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', binary=False)
    model = gensim.models.FastText.load_fasttext_format('wiki.en.bin')
elif embedding == "1":
    #model = gensim.models.FastText.load_fasttext_format('crawl-300d-2M.vec','vec')
    model = gensim.models.KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False)
elif embedding == "2":
    model = gensim.models.KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', binary=False, limit = 50000)
# test = ["computer-science", "computer_science", "Computer-Science", "computer science", "Computer_Science", "Computer Science"]
# for n in test:
#     if n in model.wv:
#         print "Compound is being done by :" + n

# print model.wv.vocab
# valid_words, compound_words = valid_words(relations, model)
# print compound_words
outliers = []
if mode == "test":
    gold, relations, taxonomy = read_all_data()
    for i in range(1,10):
        print(len(relations))
        outliers = calculate_outliers(relations,model, embedding)
        relations = compare_to_gold(gold, relations,  outliers, model)

elif mode =="train":
    gold, relations, taxonomy = read_trial_data()
    outliers = calculate_outliers(relations, model, embedding)
    compare_to_gold(gold, taxonomy, outliers)

elif mode =="train_embeddings":
    model = gensim.models.FastText.load("trial_emb")
    print(model.wv.similarity("dirty", "clean"))
    # gold,relations,taxonomy = read_all_data()
    # vocabulary = [relation[2] for relation in relations] + [relation[1] for relation in relations]
    # documents = list(read_input(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews_data.txt.gz" ),vocabulary))
    # model = gensim.models.FastText(size= 300, min_count = 2, workers = 30)
    # model.build_vocab(documents)
    # #model.train(documents, total_examples = len(documents), epochs=10)
    # model.train(documents, total_examples=model.corpus_count, epochs=2)
    # model.save("trial_emb")

elif mode =="plot_test":
    threshholds = range(2,8)
    threshholds = [float(value / 10) for value in threshholds]
    for value in threshholds:
        gold, relations, taxonomy = read_all_data()
        outliers = calculate_outliers(relations,model, embedding, value)
        compare_to_gold(gold, taxonomy, outliers, True, "outlier_removal_fasttext_check", value)

elif mode =="plot_trial":
    threshholds = range(2,8)
    threshholds = [float(value / 10) for value in threshholds]
    for value in threshholds:
        gold, relations, taxonomy = read_trial_data()
        outliers = calculate_outliers(relations,model, embedding, value)
        compare_to_gold(gold, taxonomy, outliers, True, "outlier_removal_fasttext_check", value)

elif mode =="gridsearch_test_iterative":
    threshholds = range(10000, 100000, 10000)
    threshholds = [float(value / 10) for value in threshholds]
    for value in threshholds:
        gold, relations, taxonomy = read_all_data()
        for i in range(1,5):
            outliers = calculate_outliers(relations,model, embedding, value)
            relations = compare_to_gold(gold, relations, outliers, model, True, "logs/wikipedia_2M_outlier_removal_by_rank_adding_back__by_rank_iterative_5/", value)



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
