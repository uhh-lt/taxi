from sklearn.metrics.pairwise import cosine_similarity
import sys
import csv
import numpy as np

filename_in = None
word_emebeddings = None
embedding_dim = 300

if len(sys.argv) >= 2:
    filename_in = sys.argv[1]

if len(sys.argv) >= 3:
    word_embeddings = sys.argv[2]

if len(sys.argv) >= 4:
    embedding_dim = sys.argv[3]

list_data = []
with open(filename_in, 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data.append((line[0], line[1], line[2]))

taxonomy_vocabulary = []

for relation in list_data:
    if relation[0] not in taxonomy_vocabulary:
        taxonomy_vocabulary.append(relation[0])
    if relation[1] not in taxonomy_vocabulary:
        taxonomy_vocabulary.append(relation[1])

print taxonomy_vocabulary

embeddings = {}

with open(word_embeddings, 'rb') as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in xrange(vocab_size):
        word = []
        while True:
            ch = f.read(1)
            if ch == ' ':
                word = ''.join(word)
                break
            if ch != '\n':
                word.append(ch)
        embeddings[word] = np.fromstring(f.read(binary_len), dtype='float32')


taxonomy_embedding = {}
for word in taxonomy_vocabulary:
    if word in embeddings:
        taxonomy_embedding[word] = embeddings[word]

print len(taxonomy_embedding)



#
# for line in f:
#     values = line.split()
#     try:
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     except ValueError:
#         continue
#
# embedding_matrix = {}
# for i in range(len(taxonomy_vocabulary)):
#     word = taxonomy_vocabulary[i]
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[word] = embedding_vector

#print len(embedding_matrix)
