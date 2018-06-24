from sklearn.metrics.pairwise import cosine_similarity
import sys
import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from scipy import spatial


def read_embeddings(embedding_path):
    embeddings = {}

    with open(embedding_path, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
        #for line in xrange(10000):
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

def taxonomy_embeddings(relations, word_embeddings):
    taxonomy_vocabulary = []

    for relation in relations:
        if relation[1] not in taxonomy_vocabulary:
            taxonomy_vocabulary.append(relation[1])
        if relation[2] not in taxonomy_vocabulary:
            taxonomy_vocabulary.append(relation[2])

    embeddings = word_embeddings

    taxonomy_embedding = {}
    for word in taxonomy_vocabulary:
        if word in embeddings:
            taxonomy_embedding[word] = embeddings[word]

    taxonomy_list = []
    for key,value in taxonomy_embedding.iteritems():
        taxonomy_list.append(value)

    return taxonomy_embedding


def taxonomy_cossine(taxonomy_embedding, relations, text):
    similarity_list = []
    min_v = 10000
    min_element = None
    for relation in relations:
        if relation[1] in taxonomy_embedding.keys() and relation[2] in taxonomy_embedding.keys():
            similarity = spatial.distance.cosine(taxonomy_embedding[relation[1]],taxonomy_embedding[relation[2]])
            # if min_v > similarity:
            #     min_v = similarity
            #     min_element = relation[1] + " " + relation[2]
            similarity_list.append(similarity)
            #similarity_true.append(cosine_true[words_true.index(relation[1])][words_true.index(relation[2])])
    # plt.plot(similarity_true)
    # plt.show()
    print text + " relationship properties:"
    print "============================="
    print "Average: " + str(sum(similarity_list)/float(len(similarity_list)))
    #print "Min: " + str(min(similarity_list))
    #print min_element + '\n'


def k_closest_words(k, taxonomy_embedding, relations, text):
    true_count = 0
    neighbors = []
    for word in taxonomy_embedding.keys():
        distances = []
        dict_dist = {}
        for neighbor in taxonomy_embedding.keys():
            if word != neighbor:
                distance = spatial.distance.cosine(taxonomy_embedding[word],taxonomy_embedding[neighbor])
                dict_dist[distance] = neighbor
                distances.append(distance)

        distances.sort()
        k_nearest = distances[:k]
        word_relations = []
        for relation in relations:
            if relation[1] == word:
                word_relations.append(relation[2])
            if relation[2] == word:
                word_relations.append(relation[1])
        relation_neighbors = [neighbor for neighbor in k_nearest if dict_dist[neighbor] in word_relations]
        if relation_neighbors:
            for rel in relation_neighbors:
                if distance > 0.9:
                    print word + " " + dict_dist[neighbor]
            true_count+=1
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
                if child1 != child2 and not (child1, child2) in [(x2, x3) for (x1,x2,x3) in relations]:
                    relations.append((id, child1, child2))
                    id+=1
    return relations



def get_lower_bound(word_embeddings, filename_in = None):
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

    if filename_in != "None":
        list_data = []
        with open(filename_in, 'rb') as f:
            reader = csv.reader(f, delimiter = '\t')
            for i, line in enumerate(reader):
                list_data.append((line[0], line[1], line[2]))
        taxonomy_embedding= taxonomy_embeddings(list_data, word_embeddings)
        #taxonomy_cossine(taxonomy_embedding, list_data, "Taxonomy")
        vectors= []
        for key, value in taxonomy_embedding.iteritems():
            vectors.append(value)
        hypernyms = co_hypernymy(list_data)
        #visualize_taxonomy(vectors, taxonomy_embedding.keys())
        k_closest_words(5, taxonomy_embedding,co_hypernyms_relations(hypernyms), "Taxonomy")



    taxonomy_embedding_true = taxonomy_embeddings(correct_relations, word_embeddings)
    taxonomy_embedding_false = taxonomy_embeddings(wrong_relations, word_embeddings)

    # cosine_true = cossine_matrix(vectors_true)
    # cosine_false = cossine_matrix(vectors_false)


    #k_closest_words(10, taxonomy_embedding_false, wrong_relations, "Wrong")
    #k_closest_words(10, taxonomy_embedding_true, correct_relations, "True")
    # taxonomy_cossine(taxonomy_embedding_true, correct_relations, "True")
    # taxonomy_cossine(taxonomy_embedding_false, wrong_relations, "Wrong")





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

def remove_outliers_taxonomy(filename_in, threshhold):
    list_data = []
    with open(filename_in, 'rb') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            list_data.append((line[0], line[1], line[2]))


filename_in = None
word_embeddings = None

if len(sys.argv) >= 2:
    filename_in = sys.argv[1]

if len(sys.argv) >= 3:
    word_embeddings = sys.argv[2]



embeddings = read_embeddings(word_embeddings)
get_lower_bound(embeddings, filename_in)


# remove_outliers_taxonomy(filename_in, threshhold)


#visualize_taxonomy(taxonomy_list, taxonomy_vocabulary.keys())
