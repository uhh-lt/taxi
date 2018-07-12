from sklearn.metrics.pairwise import cosine_similarity
import sys
import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from scipy import spatial
import sense2vec


def read_embeddings(embedding_path):
    embeddings = {}

    with open(embedding_path, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
        #for line in xrange(100000):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            embeddings[word] = np.fromstring(f.read(binary_len), dtype='float32')
    print len(embeddings)
    return embeddings

def taxonomy_embeddings(taxonomy_vocabulary, word_embeddings):

    print len(taxonomy_vocabulary)

    taxonomy_embedding = {}
    for word in taxonomy_vocabulary:
        if word in embeddings:
            taxonomy_embedding[word] = embeddings[word]
        #if word is compound word of words that are found in embedding take weighted average
        elif set(word.split('_')).issubset(set(embeddings.keys())):
            #print word
            word_parts = word.split('_')
            compound_word = embeddings[word_parts[0]]
            for i in range(1, len(word_parts)):
                part = word_parts[i]
                compound_word += embeddings[part]
            #compound_word /= (len(word_parts))
            taxonomy_embedding[word] = compound_word

    taxonomy_list = []
    for key,value in taxonomy_embedding.iteritems():
        taxonomy_list.append(value)

    return taxonomy_embedding


#Calculates average and tries to find upper and lower bounds for relations regarding their cos-distance or similarity
def relation_distance(taxonomy_embedding, relations, text):
    distance_list = []
    distance_list_inv = {}
    max_v = 0
    max_element = None
    for relation in relations:
        if relation[1] in taxonomy_embedding.keys() and relation[2] in taxonomy_embedding.keys():
            distance = spatial.distance.cosine(taxonomy_embedding[relation[1]],taxonomy_embedding[relation[2]])
            if max_v < distance:
                max_v = similarity
                max_element = relation[1] + " " + relation[2]
            distance_list.append(distance)
            distance_list_inv[distance] = relation
            #similarity_true.append(cosine_true[words_true.index(relation[1])][words_true.index(relation[2])])
    # plt.plot(similarity_true)
    # plt.show()
    print text + " relationship properties:"
    print "============================="
    print "Average: " + str(sum(distance_list)/float(len(distance_list)))
    print "Max: " + str(min(distance_list))
    print max_element + '\n'
    return distance_list, distance_list_inv


def k_closest_words(k, taxonomy_embedding, relations, text):
    outliers = {}
    true_count = 0
    neighbors = []
    print len(relations)
    for word in taxonomy_embedding.keys():
        distances = []
        dict_dist = {}
        for neighbor in taxonomy_embedding.keys():
            if word != neighbor:
                distance = spatial.distance.cosine(taxonomy_embedding[word],taxonomy_embedding[neighbor])
                dict_dist[distance] = neighbor
                distances.append(distance)

        distances.sort()
        #print distances
        dict_dist_sort = []
        for element in distances:
            dict_dist_sort.append(dict_dist[element])
        #print word
        #print dict_dist_sort
        k_nearest = distances[:k]
        word_relations = set([])
        #word_relations = co-hypernyms[word]
        for relation in relations:
            # if relation[1] == word:
            #     word_relations.add(relation[2])
            if relation[2] == word:
                word_relations.add(relation[1])
        relation_neighbors = [neighbor for neighbor in distances if dict_dist[neighbor] in word_relations]
        if relation_neighbors:
            average =  0
            for rel in relation_neighbors:
                average += rel
            average /= len(relation_neighbors)
            #print average
            for rel in relation_neighbors:
                if rel > 0.60:
                    if dict_dist[rel] in outliers:
                        outliers[dict_dist[rel]] +=1
                    else:
                        outliers[dict_dist[rel]] = 1
                    print word + " " + dict_dist[rel]
            true_count+=1
    sorted_outliers = sorted(outliers.items(), key=lambda x: x[1])
    print sorted_outliers
    return [element[0] for element in sorted_outliers if element[1] >= 1]
    #print outliers

    print text + str(true_count / float(len(taxonomy_embedding.keys())))

def co_hypernymy(relations):
    structure = {}
    for parent in [relation[2] for relation in relations]:
        structure[parent] = [relation[1] for relation in relations if relation[2] == parent]
    return structure

def co_hypernyms_relations(co_hypernyms):
    relations = []
    id = 0
    for hypernym, children in co_hypernyms.iteritems():
        for child1 in children:
            for child2 in children:
                if child1 != child2:
                    relations.append((id, child1, child2))
                    id+=1
        print id
    return relations


def read_trial_data():
    all_info = []
    trial_dataset_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../resources","relations.csv")
    with open(trial_dataset_fpath, 'rb') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            #id, hyponym, hypernym, correct, source
            all_info.append((line[0], line[1], line[2], line[3], line[4]))
    correct_relations = []
    wrong_relations = []
    for entry in all_info:
        if entry[4] in ["WN_plants.taxo", "WN_vehicles.taxo", "ontolearn_AI.taxo", "negative co-hypo"]: #alternatively add negative co-hypo
            if entry[3] == "1":
                correct_relations.append((entry[0], entry[1], entry[2]))
        else:
            wrong_relations.append((entry[0], entry[1], entry[2]))

    return [correct_relations, wrong_relations]


def taxonomy_eval(filename_in, filename_gold, word_embeddings, filename_out, mode = 'visualize'):
    list_data_o = []
    list_data = []
    with open(filename_in, 'rb') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            list_data.append((line[0], line[1], line[2]))
            list_data_o.append((line[0], line[1], line[2]))

    #compound words in taxonomy
    for i in range(len(list_data)):
        list_data[i] = (list_data[i][0], list_data[i][1].replace(" ", "_"), list_data[i][2].replace(" ", "_"))

    taxonomy_vocabulary = []

    for relation in list_data:
        if relation[1] not in taxonomy_vocabulary:
            taxonomy_vocabulary.append(relation[1])
        if relation[2] not in taxonomy_vocabulary:
            taxonomy_vocabulary.append(relation[2])


    taxonomy_embedding= taxonomy_embeddings(taxonomy_vocabulary, word_embeddings)

    # for taxonomy_embedding
    #taxonomy_cossine(taxonomy_embedding, list_data, "Taxonomy")

    if mode == 'k-nearest':
        hypernyms = co_hypernymy(list_data)
        #outliers = k_closest_words(200, taxonomy_embedding,co_hypernyms_relations(hypernyms), "Taxonomy")
        outliers = k_closest_words(200, taxonomy_embedding, list_data, "Taxonomy")
        remove_outliers_and_compare(filename_gold,list_data_o,outliers, 0.0)
        #remove_outliers_and_write(filename_out,list_data_o,outliers, 0.0)

    elif mode =='visualize':
        hypernyms = co_hypernymy(list_data)
        science_hypo = hypernyms["science"]
        science_hypo.append("science")
        print science_hypo
        science_embedding = taxonomy_embeddings(science_hypo, word_embeddings)
        science_hypos = []
        for word, value in science_embedding.iteritems():
            science_hypos.append(value)
        visualize_taxonomy(science_hypos, science_embedding.keys())
        # vectors= []
        # for key, value in taxonomy_embedding.iteritems():
        #     vectors.append(value)
        # visualize_taxonomy(vectors, taxonomy_embedding.keys())


def remove_outliers_and_compare(filename_gold, taxonomy, outliers,fraction):
    gold = []
    with open(filename_gold, 'rb') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            gold.append((line[0], line[1], line[2]))
    removed_outliers = []
    for element in taxonomy:
        # if element[1].replace(' ', '_') in outliers:
        #     print "skip: " + element[1] + " " + element[2]
        #     continue
        removed_outliers.append(element)

    correct = 0
    for element in removed_outliers:
        for ele_g in gold:
            if element[1] == ele_g[1] and element[2] == ele_g[2]:
                correct+=1
    precision = correct / float(len(removed_outliers))
    recall = correct / float(len(gold))
    print float(len(removed_outliers))
    print float(len(gold))
    print "Correct: " + str(correct)
    print "Precision: " + str(precision)
    print "Recall: " + str(recall)
    print "F1: " + str(2*precision *recall / (precision + recall))


def remove_outliers_and_write(filename_out, taxonomy, outliers,fraction):
    outliers = outliers[int(len(outliers) * fraction):]
    print outliers
    print len(taxonomy)
    with open(filename_out, 'w') as f:
        for element in taxonomy:
            if element[1].replace(' ', '_') in outliers:
                print "skip: " + element[1] + " " + element[2]
                continue
            f.write(element[0] + '\t' + element[1] + '\t' + element[2]  + '\n')
    f.close()

def trial_eval(word_embeddings, mode):

    correct_relations, wrong_relations = read_trial_data()

    taxonomy_vocabulary_true = []
    taxonomy_vocabulary_false = []

    for relation in correct_relations:
        if relation[1] not in taxonomy_vocabulary_true:
            taxonomy_vocabulary_true.append(relation[1])
        if relation[2] not in taxonomy_vocabulary_true:
            taxonomy_vocabulary_true.append(relation[2])
    for relation in wrong_relations:
        if relation[1] not in taxonomy_vocabulary_false:
            taxonomy_vocabulary_false.append(relation[1])
        if relation[2] not in taxonomy_vocabulary_false:
            taxonomy_vocabulary_false.append(relation[2])



    taxonomy_embedding_true = taxonomy_embeddings(taxonomy_vocabulary_true, word_embeddings)
    taxonomy_embedding_false = taxonomy_embeddings(taxonomy_vocabulary_false, word_embeddings)

    hypernyms_correct = co_hypernymy(correct_relations)
    hypernyms_wrong = co_hypernymy(wrong_relations)

    # cosine_true = cossine_matrix(vectors_true)
    # cosine_false = cossine_matrix(vectors_false)

    if mode == 'k-nearest':
        k_closest_words(5, taxonomy_embedding_false, co_hypernyms_relations(hypernyms_wrong), "Wrong")
        k_closest_words(10, taxonomy_embedding_true,co_hypernyms_relations(hypernyms_correct), "True")

    elif mode == 'distance':
        relation_distance(taxonomy_embedding_true, correct_relations, "True")
        relation_distance(taxonomy_embedding_false, wrong_relations, "Wrong")


def main(word_embeddings, filename_in = None, filename_gold = None, filename_out = None,  mode = 'visualize'):

    if filename_in != "None":
         taxonomy_eval(filename_in, filename_gold, word_embeddings, filename_out,  mode)


    # if filename_gold != "None":
    #     taxonomy_eval(filename_gold, word_embeddings, filename_out, mode)

    #trial_eval(word_embeddings, mode)



def visualize_taxonomy(taxonomy_vectors, taxonomy_names):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(taxonomy_vectors)

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(taxonomy_names, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def cossine_matrix(taxonomy_vectors):
    return cosine_similarity(taxonomy_vectors, taxonomy_vectors)

def find_matches(taxonomy, s2v):
    words = set([])
    for relation in taxonomy:
        words.add(relation[1])
        words.add(relation[2])
    pairs = []
    for word in words:
        for word2 in words:
            if not (word, word2) in pairs:
                pairs.append((word, word2))
    for element in  pairs:
        if element[0].decode('unicode-escape') + '|NOUN' in s2v:
            freq, vector = s2v[element[0].decode('unicode-escape') + '|NOUN']
            most_similar, scores = s2v.most_similar(vector, n=2)
            if (element[1].decode('unicode-escape') + '|NOUN') in s2v and (element[1].decode('unicode-escape') + '|NOUN') in most_similar[0] and element[1] != element[0]:
                index = most_similar[0].index(element[1].decode('unicode-escape') + '|NOUN')
                print element[0] +  " " + element[1] + " "  + str(scores[index])

def testing_sv2(filename_in, s2v):
    list_data = []
    list_data_o = []
    with open(filename_in, 'rb') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            list_data.append((line[0], line[1], line[2]))
            list_data_o.append((line[0], line[1], line[2]))

    for i in range(len(list_data)):
        list_data[i] = (list_data[i][0], list_data[i][1].replace(" ", "_"), list_data[i][2].replace(" ", "_"))

    find_matches(list_data, s2v)

    hypernyms = co_hypernymy(list_data)
    outliers = {}
    for element in  co_hypernyms_relations(hypernyms):
        if element[1].decode('unicode-escape') + '|NOUN' in s2v:
            freq, vector = s2v[element[1].decode('unicode-escape') + '|NOUN']
            most_similar = s2v.most_similar(vector, n=10000)
            if (element[2].decode('unicode-escape') + '|NOUN') in s2v and not (element[2].decode('unicode-escape') + '|NOUN') in most_similar[0]:
                if element[2] in outliers:
                    outliers[element[2]]+=1
                else:
                    outliers[element[2]] = 1
                if element[1] in outliers:
                    outliers[element[1]]+=1
                else:
                    outliers[element[1]] = 1

                #print element[1] + " " + element[2]
            #else:
                #print "YAAY"
    outliers = sorted(outliers.items(), key=lambda x: x[1])
    print outliers
    outliers = [element[0] for element in outliers if element[1] > 20]

filename_in = None
filename_gold = None
filename_out  = None
word_embeddings = None
mode = None

if len(sys.argv) >= 2:
    filename_in = sys.argv[1]

if len(sys.argv) >= 3:
    filename_gold = sys.argv[2]

# if len(sys.argv) >= 4:
#     filename_out = sys.argv[3]

if len(sys.argv) >= 4:
    word_embeddings = sys.argv[3]


if len(sys.argv) >= 5:
    mode = sys.argv[4]
print mode
#

#s2v = sense2vec.load('reddit_vectors-1.1.0')

list_data_o = []
outliers = []
with open(filename_in, 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data_o.append((line[0], line[1], line[2]))

#taxonomy_eval(filename_in, )

# remove_outliers_and_compare(filename_gold, list_data_o, outliers,0.0)


embeddings = read_embeddings(word_embeddings)
main(embeddings, filename_in, filename_gold, mode)


# COMMAND : python embeddings.py ../out/science_en.csv-relations.csv-taxo-knn1.csv-pruned.csv-cleaned.csv ../eval/taxi_eval_archive/input/gold.taxo GoogleNews-vectors-negative300.bin k-nearest
