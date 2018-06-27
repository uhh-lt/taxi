import gensim

filename_in = "science_en.csv-relations.csv-taxo-knn1.csv-pruned.csv-cleaned.csv"
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
list_data_o = []
with open(filename_in, 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data_o.append((line[0], line[1], line[2]))

print model.wv.doesnt_match(["cat","dog","france"])
