import codecs
import ntpath
from random import random
from time import time

import numpy as np
from pandas import read_csv, Series, merge, concat

from jnt.common import exists
from jnt.isas.isas import ISAs
from jnt.freq import FreqDictionary
from jnt.isas.taxo import TaxonomyResources

MAX_PROXY_ISAS = 10

def taxo2csv_all_correct(taxo_fpath):
    taxo = read_csv(taxo_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    dataset_name = ntpath.basename(taxo_fpath)
    taxo["correct"] = Series(np.ones(len(taxo)), index=taxo.index)

    output_fpath = taxo_fpath + ".csv"
    taxo.to_csv(output_fpath, sep="\t", encoding="utf-8", float_format='%.0f', index=False)
    
    print(output_fpath)    
    return taxo


def taxo2csv_mixed(taxo_fpath, taxo_eval_fpath):
    output_fpath = taxo_fpath + ".csv"

    taxo = read_csv(taxo_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    taxo_eval = read_csv(taxo_eval_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)

    result = merge(taxo, taxo_eval, on='relation_id')
    print(output_fpath + "-tmp.csv")
    result = result.fillna(1)
    result = result.replace("x", 0, regex=False)
    result.to_csv(output_fpath + "-tmp.csv", sep="\t", encoding="utf-8", float_format='%.0f', index=False)
    
    with codecs.open(output_fpath + "", "w", "utf-8") as output:
        print("relation_id\thyponym\thypernym\tcorrect", file=output)
        for i, row in result.iterrows():
            hyponyms = str(row.hyponym).split(",")
            hypernyms = str(row.hypernym).split(",")
            for hyponym in hyponyms:
                for hypernym in hypernyms:
                    print("%s\t%s\t%s\t%s" % (row.relation_id, hyponym, hypernym, row.correct), file=output)
    
    df = read_csv(output_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    print(output_fpath)
    return df

    
def insert_source(fpath, relations):
    dataset_name = ntpath.basename(fpath)
    relations["source"] = Series([dataset_name for x in range(len(relations))], index=relations.index)
    return relations
    
    
def add_inverse_relations(relations):
    for i, row in relations.iterrows():
        if row.correct == 1:  # invert only positive relations
            relations.loc[len(relations)] = [len(relations), row.hypernym, row.hyponym, 0,  row.source]
        relations = relations.sort_values(["hyponym", "correct"], ascending=[1,0])
    
    return relations


def add_cohypo_negatives(relations, isa_fpath):
    taxo_res = TaxonomyResources(freq_fpaths=[""], isa_fpaths=[isa_fpath])
    isas = taxo_res.isas[list(taxo_res.isas.keys())[0]]

    neg_num = 0
    for hyper in isas.data:
        hypos = [word for word, freq in isas.all_hypo(hyper)]
        if len(hypos) > 1:
            print(hyper.upper(), len(hypos))

            for hypo1 in hypos:
                for hypo2 in hypos:
                    if hypo1 == hypo2: continue
                    relations.loc[len(relations)] = [len(relations), hypo1, hypo2, 0,  "negative co-hypo"]
                    neg_num += 1
    relations = relations.sort_values(["hyponym", "correct"], ascending=[1,0])
    print("Added %d negative co-hypo relations" % neg_num)
    return relations


def remove_underscores(relations):
    for i, row in relations.iterrows():
        relations.loc[i,"hyponym"] = row.hyponym.replace("_", " ")
        relations.loc[i,"hypernym"] = row.hypernym.replace("_", " ")    
    return relations

   
def fill_frequencies(freq_fpath, relations, field_name_postfix=""):
    freq = FreqDictionary(freq_fpath)

    hyponym_freq = np.zeros(len(relations))
    hypernym_freq = np.zeros(len(relations))

    for i, row in relations.iterrows():
        hyponym_freq[i] = freq.freq(row.hyponym)
        hypernym_freq[i] = freq.freq(row.hypernym)
    relations["hyponym_freq" + field_name_postfix] = Series(hyponym_freq, index=relations.index)
    relations["hypernym_freq" + field_name_postfix] = Series(hypernym_freq, index=relations.index)
    
    return relations   


def str_in_str(str1, str2):
    if str1 in str2:
        return float(len(str1)) / float(len(str2)) 
    elif len(str1) >= 7 and str1[:-3] in str2:
        return float(len(str1)-3) / float(len(str2)) 
    else:
        return 0.0


def fill_substrings(relations):
    hyper_in_hypo = np.zeros(len(relations))
    hypo_in_hyper = np.zeros(len(relations))

    for i, row in relations.iterrows():
        hyper_in_hypo[i] = str_in_str(row.hypernym, row.hyponym)
        hypo_in_hyper[i] = str_in_str(row.hyponym, row.hypernym)
    relations["hyper_in_hypo"] = Series(hyper_in_hypo, index=relations.index)
    relations["hypo_in_hyper"] = Series(hypo_in_hyper, index=relations.index)
    
    return relations 


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


def predict_by_isas(relations, field_name_postfix=""):
    
    isa_col = "isa_freq" + field_name_postfix
    ais_col = "ais_freq" + field_name_postfix
    
    predict = np.zeros(len(relations))
    for i, row in relations.iterrows():
        predict[i] = row[isa_col] > row[ais_col]
        
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
        predict[i] = row["hyper_in_hypo"] > row["hypo_in_hyper"]
    
    relations["correct_predict"] = Series(predict, index=relations.index)
    return relations



def fill_degrees(isas_fpath, relations, field_name_postfix=""):
    isas = ISAs(isas_fpath)
    relations = fill_in_degrees(isas, relations, field_name_postfix)
    relations = fill_out_degrees(isas, relations, field_name_postfix)

    return relations


def fill_in_degrees(isas, relations, field_name_postfix=""):
    hypo_in_num = np.zeros(len(relations))
    hyper_in_num = np.zeros(len(relations))
    hypo_in_weight = np.zeros(len(relations))
    hyper_in_weight = np.zeros(len(relations))

    for i, row in relations.iterrows():
        hypo_hypos = isas.all_hypo(row.hyponym)
        hyper_hypos = isas.all_hypo(row.hypernym)
        hypo_in_num[i] = len(hypo_hypos)
        hyper_in_num[i] = len(hyper_hypos)
        hypo_in_weight[i] = sum([freq for hypo, freq in hypo_hypos])
        hyper_in_weight[i] = sum([freq for hypo, freq in hyper_hypos])

    relations["hypo_in_num" + field_name_postfix] = Series(hypo_in_num, index=relations.index)
    relations["hyper_in_num" + field_name_postfix] = Series(hyper_in_num, index=relations.index)
    relations["hypo_in_weight" + field_name_postfix] = Series(hypo_in_weight, index=relations.index)
    relations["hyper_in_weight" + field_name_postfix] = Series(hyper_in_weight, index=relations.index)

    return relations


def fill_out_degrees(isas, relations, field_name_postfix=""):
    hypo_out_num = np.zeros(len(relations))
    hyper_out_num = np.zeros(len(relations))
    hypo_out_weight = np.zeros(len(relations))
    hyper_out_weight = np.zeros(len(relations))

    for i, row in relations.iterrows():
        hypo_hypers = isas.all_hyper(row.hyponym)
        hyper_hypers = isas.all_hyper(row.hypernym)
        hypo_out_num[i] = len(hypo_hypers)
        hyper_out_num[i] = len(hyper_hypers)
        hypo_out_weight[i] = sum([freq for hyper, freq in hypo_hypers])
        hyper_out_weight[i] = sum([freq for hyper, freq in hyper_hypers])

    relations["hypo_out_num" + field_name_postfix] = Series(hypo_out_num, index=relations.index)
    relations["hyper_out_num" + field_name_postfix] = Series(hyper_out_num, index=relations.index)
    relations["hypo_out_weight" + field_name_postfix] = Series(hypo_out_weight, index=relations.index)
    relations["hyper_out_weight" + field_name_postfix] = Series(hyper_out_weight, index=relations.index)

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

def proxy_path_mean(hypo, hyper, isas):
    direct_isas = isas.all_isas(hypo, MAX_PROXY_ISAS)

    paths = []
    path_weights = []
    if hyper not in direct_isas:
        for proxy, _ in direct_isas:
            proxy_isas_freqs = isas.all_isas(proxy, MAX_PROXY_ISAS)
            if len(proxy_isas_freqs) == 0: continue
            proxy_isas, _ = list(zip(*proxy_isas_freqs))
            proxy_isas = set(proxy_isas)

            if hyper in proxy_isas:
                path_weight = (isas.data[hypo][proxy] + isas.data[proxy][hyper])/2
                paths.append([hypo, proxy, hyper])
                path_weights.append(path_weight)
    else:
        paths.append([hypo, hyper])
        path_weights.append(isas.data[hypo][hyper])

    return paths, mean(path_weights)


def proxy_path_max(hypo, hyper, isas):
    """ strategy is 'max' or 'mean' """

    direct_isas = isas.all_hyper(hypo, MAX_PROXY_ISAS)

    best_path = []
    best_path_weight = 0
    if not hyper in direct_isas:
        for proxy, _ in direct_isas:
            proxy_isas_freqs = isas.all_hyper(proxy, MAX_PROXY_ISAS)
            if len(proxy_isas_freqs) == 0: continue
            proxy_isas, _ = list(zip(*proxy_isas_freqs))
            proxy_isas = set(proxy_isas)

            if hyper in proxy_isas:
                path_weight = (isas.data[hypo][proxy] + isas.data[proxy][hyper])/2
                if path_weight > best_path_weight:
                    best_path = [hypo, proxy, hyper]
                    best_path_weight = path_weight
    else:
        best_path = [hypo, hyper]
        best_path_weight = isas.data[hypo][hyper]
    #if len(best_path) > 0: print best_path_weight, best_path
    #else: print ".",
    return best_path, best_path_weight


def fill_isas(isas_fpath, relations, field_name_postfix="", subphrases=False, second_order=False):
    isas = ISAs(isas_fpath)

    isa_freq = np.zeros(len(relations))
    ais_freq = np.zeros(len(relations))

    for i, row in relations.iterrows():
        if i % 100 == 0: print(i, isas_fpath)
        isa_freq[i] = isas.has_isa(row.hyponym, row.hypernym)
        ais_freq[i] = isas.has_isa(row.hypernym, row.hyponym)

        if subphrases and isa_freq[i] == 0 and ais_freq[i] == 0:
            # no match --> try to find a substring match
            hypo = row.hyponym.split()
            hyper = row.hypernym.split()

            isa_candidates = []
            for j in range(len(hypo)):
                for k in range(len(hyper)):
                    hypo_cut = " ".join(hypo[j:])
                    hyper_cut = " ".join(hyper[k:])
                    if (j == 0 and k == 0) or (hypo_cut == hyper_cut): continue
                    isa_candidates.append((hypo_cut, hyper_cut))

            for hypo_cut, hyper_cut in isa_candidates:
                isa_freq_cut = isas.has_isa(hypo_cut, hyper_cut)
                ais_freq_cut = isas.has_isa(hyper_cut, hypo_cut)
                if isa_freq_cut > 0 and ais_freq_cut > 0:
                    isa_freq[i] = isa_freq_cut
                    ais_freq[i] = ais_freq_cut
                    break

        if second_order and isa_freq[i] == 0 and ais_freq[i] == 0:
            _, isa_freq[i] = proxy_path_max(row.hyponym, row.hypernym, isas)
            _, ais_freq[i] = proxy_path_max(row.hypernym, row.hyponym, isas)

    relations["isa_freq" + field_name_postfix] = Series(isa_freq, index=relations.index)
    relations["ais_freq" + field_name_postfix] = Series(ais_freq, index=relations.index)

    return relations

def fill_average_isas(relations, field_name_postfix=""):
    p = field_name_postfix
    pm_max = max(relations["isa_freq_pm"+p].max(), relations["ais_freq_pm"+p].max())
    ma_max = max(relations["isa_freq_ma"+p].max(), relations["ais_freq_ma"+p].max())
    ps_max = max(relations["isa_freq_ps"+p].max(), relations["ais_freq_ps"+p].max())
    cc_max = max(relations["isa_freq_cc"+p].max(), relations["ais_freq_cc"+p].max())

    relations["isa_freq"+p] = (
        relations["isa_freq_pm"+p] / pm_max + \
        relations["isa_freq_ma"+p] / ma_max + \
        relations["isa_freq_ps"+p] / ps_max + \
        relations["isa_freq_cc"+p] / cc_max ) / 4

    relations["ais_freq"+p] = (
        relations["ais_freq_pm"+p] / pm_max + \
        relations["ais_freq_ma"+p] / ma_max + \
        relations["ais_freq_ps"+p] / ps_max + \
        relations["ais_freq_cc"+p] / cc_max ) / 4

    return relations

def accuracy(relations, name=""):
    print("Accuracy %s: %.3f" %(name, sum(relations.correct == relations.correct_predict)/float(len(relations))))

   

########
def load_relations(relations_fpath, taxo_en_plants_fpath="", taxo_en_vehicles_fpath="", taxo_en_ai_fpath="", taxo_eval_en_ai_fpath=""):
    if exists(relations_fpath):
        relations = read_csv(relations_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
        print("Relations loaded from:", relations_fpath)

    elif exists(taxo_en_plants_fpath) and exists(taxo_en_vehicles_fpath) and exists(taxo_en_ai_fpath) and exists(taxo_eval_en_ai_fpath):
        tic = time()
        plants = taxo2csv_all_correct(taxo_en_plants_fpath)
        plants = insert_source(taxo_en_plants_fpath, plants)
        print("plants:", len(plants))

        vehicles = taxo2csv_all_correct(taxo_en_vehicles_fpath)
        vehicles = insert_source(taxo_en_vehicles_fpath, vehicles)
        print("vehicles:", len(vehicles))

        ai = taxo2csv_mixed(taxo_en_ai_fpath, taxo_eval_en_ai_fpath)
        ai = insert_source(taxo_en_ai_fpath, ai)
        print("ai:", len(ai))

        relations = concat([plants, vehicles, ai], ignore_index=True)
        print("all:", len(relations))

        relations = remove_underscores(relations)
        relations = add_inverse_relations(relations)
        relations = relations.sort_values(["hyponym", "correct"], ascending=[1,0])
        relations.to_csv(relations_fpath, sep="\t", encoding="utf-8", float_format='%.0f', index=False)
        relations = read_csv(relations_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
        print("Dataset:", relations_fpath)
        print("Relations generated and loaded in %.1f sec." % (time()-tic))
        
    else:
        print("Error: cannot load relations. No input files found.") 
        relations = None

    return relations
