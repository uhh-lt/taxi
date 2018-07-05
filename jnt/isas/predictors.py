import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from random import random
import numpy as np
from pandas import Series
from jnt.isas.supervised import SuperTaxi


class TaxonomyPredictor():
    def __init__(self, taxo_features):
        self._features = taxo_features
        self._relations = taxo_features.relations

    def evaluate(self, field=""):
        accuracy(self._relations, field)

    def get_relations():
        return self._relations
        
    def predict_and_evaluate(self):
        self.predict_by_global_threshold(threshold=0.0, field="hyper_in_hypo_i")
        self.evaluate("hyper_in_hypo_i")

        self.predict_by_global_threshold(threshold=0.0, field="hypo2hyper_substract")
        self.evaluate("hypo2hyper_substract")

        for p in ["", "2", "3", "_s", "2_s", "3_s", "_max2", "_mean2"]:
            r = predict_by_isas(self._relations, p)
            accuracy(r, "hypo2hyper" + p)

        for isa_name in self._features._isas:
            p = "_" + isa_name
            r = predict_by_isas(self._relations, p)
            accuracy(r, "hypo2hyper" + p)

            r = predict_by_out_degrees(self._relations, p, weight=False)
            accuracy(r, "out degrees num" + p)

            r = predict_by_out_degrees(self._relations, p, weight=True)
            accuracy(r, "out degrees weight" + p)

            r = predict_by_in_degrees(self._relations, p, weight=False)
            accuracy(r, "in degrees num" + p)

            r = predict_by_in_degrees(self._relations, p, weight=True)
            accuracy(r, "in degrees weight" + p)

        r = predict_by_length(self._relations)
        accuracy(r, "length")

        r = predict_by_random(self._relations)
        accuracy(r, "random")

        r = predict_by_word_freq(self._relations)
        accuracy(r, "hypo_freq")

    def save(self, taxonomy_fpath, conf=False):
        df = self._relations.copy()
        df = df[df["correct_predict"] == 1]

        if conf:
            df = df[["hyponym","hypernym","correct_predict_conf"]]
            df = df.sort_values("correct_predict_conf", ascending=0)
            df.to_csv(taxonomy_fpath, sep="\t", encoding="utf-8", float_format='%.5f', index=True, header=False)
        else:
            df = df[["hyponym","hypernym"]]
            df = df.sort_values("hyponym", ascending=1)
            df.to_csv(taxonomy_fpath, sep="\t", encoding="utf-8", float_format='%.5f', index=True, header=False)

        print("Taxonomy:", taxonomy_fpath)

    def predict_by_classifier(self, classifier_dir):
        st = SuperTaxi(classifier_dir)
        self._relations = st.predict(self._relations)
        return st.meta["method"]

    def predict_by_local_threshold(self, max_knn=3, threshold=0.0, field="", or_correct_predict=False):
        predict = np.zeros(len(self._relations))

        prev_hypo = ""
        k = 0
        df = self._relations.sort_values(["hyponym",field], ascending=[1,0])
        for i, row in df.iterrows():
            score = row[field]
            if prev_hypo != row.hyponym: k = 0
            if k < max_knn and score > threshold: predict[i] = 1

            k += 1
            prev_hypo = row.hyponym
            #if score > 0:
            #    print score, k, row.hyponym, prev_hypo, row.hypernym, predict[i]

        self._relations["tmp"] = Series(predict, index=self._relations.index, dtype=float)
        if or_correct_predict and "correct_predict" in self._relations:
            self._relations["correct_predict"] = self._relations[["tmp","correct_predict"]].max(axis=1)
        else:
            self._relations["correct_predict"] = self._relations["tmp"]

        self._relations = self._relations.drop('tmp', 1)

    def predict_by_global_threshold(self, threshold=0.0, field="", or_correct_predict=False):
        predict = np.zeros(len(self._relations))

        for i, row in self._relations.iterrows():
            score = row[field]
            if score <= threshold:
                predict[i] = 0
            else:
                predict[i] = 1

        self._relations["tmp"] = Series(predict, index=self._relations.index, dtype=float)
        if or_correct_predict and "correct_predict" in self._relations:
            self._relations["correct_predict"] = self._relations[["tmp","correct_predict"]].max(axis=1)
        else:
            self._relations["correct_predict"] = self._relations["tmp"]

        self._relations = self._relations.drop('tmp', 1)


def predict_by_length(relations):
    predict = np.zeros(len(relations))
    for i, row in relations.iterrows():
        predict[i] = len(row.hypernym) < len(row.hyponym)

    relations["correct_predict"] = Series(predict, index=relations.index)

    return relations


def predict_by_word_freq(relations, comparable_freqs_heuristic=False, field_name_postfix=""):
    hypernym_col = "hypernym_freq" + field_name_postfix
    hyponym_col = "hyponym_freq" + field_name_postfix

    predict = np.zeros(len(relations))
    for i, row in relations.iterrows():
        if comparable_freqs_heuristic:
            comparable_freqs = row[hypernym_col] / float(row[hyponym_col]) < 2
            if comparable_freqs:
                predict[i] = len(row["hyponym"]) > len(row["hypernym"])
            else:
                predict[i] = row[hypernym_col]> row[hyponym_col]
        else:
            predict[i] = row[hypernym_col] > row[hyponym_col]

    relations["correct_predict"] = Series(predict, index=relations.index)

    return relations


def predict_by_isas(relations, p=""):
    predict = np.zeros(len(relations))

    for i, row in relations.iterrows():
        hypo2hyper = row["hypo2hyper" + p]
        hyper2hypo = row["hyper2hypo" + p]
        if hypo2hyper == 0 and hyper2hypo == 0:
            predict[i] = 0
        else:
            predict[i] = hypo2hyper > hyper2hypo

    relations["correct_predict"] = Series(predict, index=relations.index)
    return relations


def predict_by_random(relations):
    predict = np.zeros(len(relations))
    for i, row in relations.iterrows():
        predict[i] = int(random() > 0.5)

    relations["correct_predict"] = Series(predict, index=relations.index)
    return relations


def predict_by_substrings(relations):
    predict = np.zeros(len(relations))
    for i, row in relations.iterrows():
        hypo_in_hyper = row["hypo_in_hyper"]
        hyper_in_hypo = row["hyper_in_hypo"]
        if hypo_in_hyper == 0 and hyper_in_hypo == 0:
            predict[i] = 0
        else:
            predict[i] = hypo_in_hyper == 0 and hyper_in_hypo > 0

    relations["correct_predict"] = Series(predict, index=relations.index)
    return relations


def predict_by_out_degrees(relations, field_name_postfix="", weight=False):
    score = "weight" if weight else "num"
    hypo_col = "hypo_out_" + score + field_name_postfix
    hyper_col = "hyper_out_" + score + field_name_postfix

    predict = np.zeros(len(relations))
    for i, row in relations.iterrows():
        predict[i] = row[hypo_col] < row[hyper_col]

    relations["correct_predict"] = Series(predict, index=relations.index)

    return relations


def predict_by_in_degrees(relations, field_name_postfix="", weight=False):
    score = "weight" if weight else "num"
    hypo_col = "hypo_in_" + score + field_name_postfix
    hyper_col = "hyper_in_" + score + field_name_postfix

    predict = np.zeros(len(relations))
    for i, row in relations.iterrows():
        predict[i] = row[hypo_col] < row[hyper_col]

    relations["correct_predict"] = Series(predict, index=relations.index)

    return relations


def accuracy(relations, name=""):
    print("\n", name.upper(), "\n", "="*50)
    print("correct:", sum(relations.correct == relations.correct_predict))
    print("all:", len(relations))
    print("accuracy: %.3f" % (sum(relations.correct == relations.correct_predict)/float(len(relations))))
    print(classification_report(relations.correct, relations.correct_predict))

    try:
        precision, recall, thresholds = precision_recall_curve(relations.correct, relations[name])
        plt.clf()
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision_score(relations.correct, relations.correct_predict)))
        plt.legend(loc="lower left")
        plt.show()
    except:
        print("Warning: cannot make plot.")
        # print format_exc()
