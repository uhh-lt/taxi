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

    return taxonomy_list, taxonomy_embedding.keys()



def get_lower_bound(word_embeddings):
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
        if entry[4] in ["WN_plants.taxo", "WN_vehicles.taxo", "ontolearn_AI.taxo"]: #alternatively add negative co-hypo
            if entry[3] == "1":
                correct_relations.append((entry[0], entry[1], entry[2]))
            else:
                wrong_relations.append((entry[0], entry[1], entry[2]))

    print correct_relations
    vectors_true, words_true = taxonomy_embeddings(correct_relations, word_embeddings)
    vectors_false, words_wrong = taxonomy_embeddings(wrong_relations, word_embeddings)

    cosine_true = cossine_matrix(vectors_true)
    cosine_false = cossine_matrix(vectors_false)

    similarity_true = []
    for relation in correct_relations:
        if relation[1] in words_true and relation[2] in words_true:
            #similarity_true.append(1 - spatial.distance.cosine(vectors_true[words_true.index(relation[1])],vectors_true[words_true.index(relation[2])]))
            similarity_true.append(cosine_true[words_true.index(relation[1])][words_true.index(relation[2])])
    # plt.plot(similarity_true)
    # plt.show()
    print sum(similarity_true)/float(len(similarity_true))

    similarity_false = []
    for relation in wrong_relations:
        if relation[1] in words_wrong and relation[2] in words_wrong:
            similarity_false.append(cosine_false[words_wrong.index(relation[1])][words_wrong.index(relation[2])])
    print sum(similarity_false)/float(len(similarity_false))
    # plt.plot(similarity_false)
    # plt.show()


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
get_lower_bound(embeddings)


# remove_outliers_taxonomy(filename_in, threshhold)


#visualize_taxonomy(taxonomy_list, taxonomy_vocabulary.keys())
