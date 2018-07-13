import pandas as pd

fpath_relations = "/home/panchenko/joint/data/isas/semeval/super/new-relations-en1.csv"

pd.set_option('display.max_rows', 9999)
df = taxo_features.relations.copy()
df = df.sort_values("hypo2hyper_substract",ascending=0)
df = df[["hyponym","hypernym","correct","hypo2hyper"]]
df = df[df["correct"]== 0]
for correct_index in [71237, 13854, 67805, 32292, 55046, 12818, 5686, 112564, 54934, 16617,
                      31863, 54961, 45171, 38122, 49450, 4818, 23633, 526, 74034, 4820,
                      21006, 33426, 3974, 106224, 13883, 45150, 13989, 68145, 482, 69591,
                      5268, 34247, 112116, 67116, 64349, 14841, 36060, 118507, 55460,
                      43795, 43799, 28423, 22850, 32475, 66074, 9273, 62140, 49444, 17722,
                      70132, 14815, 60586, 21505, 36220, 112794, 78693, 51318, 5714,
                      106097, 22975, 17444, 60375]:
    taxo_features.relations.loc[correct_index, "correct"] = 1
taxo_features.relations.to_csv(fpath_relations, sep="\t", encoding="utf-8", float_format='%.5f', index=False)
print(fpath_relations)



# add relations upper in the hierarchy
fpath = "/home/panchenko/joint/data/isas/semeval/data/en_trial.csv"
# r = taxo_features.relations.copy()
# r = r[r["correct"] == 1]
# r = r[["hyponym","hypernym","correct"]]
# add freq column as a copy of correct
# r.to_csv(fpath, sep="\t", encoding="utf-8", float_format='%.0f', index=False)

# load taxonomy as a resource
t = TaxonomyResources(freq_fpaths=[], isa_fpaths=[fpath])
isas = t._isas["en_trial"].data

isas2 = isas.copy()
isas3 = isas.copy()

for hypo in isas:
    hypers = isas[hypo]
    print(hypo, list(hypers.keys()))
    for hyper in hypers:
        pass
        print("\t", hypo, "-->", hyper , "-->", list(isas2[hyper].keys()))

        hypers_hypers = isas3[hyper]
        for hyper_hyper in hypers_hypers:
            print("\t\t", hyper_hyper, isas3[hyper_hyper])

#r[hypo,hyper,correct,correct_new]