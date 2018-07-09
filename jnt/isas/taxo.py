from re import split
from collections import defaultdict
import codecs
from time import time
import numpy as np
from pandas import read_csv, Series, merge, concat
from traceback import format_exc
from numpy import mean
import operator
import re
from traceback import format_exc

from .isas import ISAs
from .predictors import *
from jnt.freq import FreqDictionary
from jnt.common import fpath2filename, try_remove, exists
from jnt.morph import lemmatize, load_stoplist


MAX_PROXY_ISAS = 10
VERBOSE = False
re_dash = re.compile(r"\s*-\s*", re.U|re.I)


class TaxonomyResources():
    def __init__(self, freq_fpaths=[], isa_fpaths=[]):

        tic = time()
        self._freqs = {}
        for fpath in freq_fpaths:
            fname = fpath2filename(fpath)
            self._freqs[fname] = FreqDictionary(fpath)
            print("Loaded freq dictionary '%s': %s" % (fname, fpath))

        self._isas = {}
        for fpath in isa_fpaths:
            fname = fpath2filename(fpath)
            self._isas[fname] = ISAs(fpath)
            print("Loaded isa dictionary (%d words) '%s': %s" % (len(self._isas[fname].data), fname, fpath))

        print("Loaded resources in %d sec." % (time() - tic))

        # load ddts here as well

    @property
    def isas(self):
        return self._isas

    @property
    def freqs(self):
        return self._freqs


class TaxonomyFeatures():
    def __init__(self, taxonomy_resources, voc_fpath="", relations_fpath="", lang="en"):
        self._isas = taxonomy_resources.isas
        self._freqs = taxonomy_resources.freqs
        self.voc_name = fpath2filename(voc_fpath)
        self._voc_fpath = voc_fpath
        self._stopwords = load_stoplist(lang=lang)
        self._lang = lang

        if exists(voc_fpath) and not exists(relations_fpath):
            self.voc = self._load_voc(voc_fpath) 
            relations_fpath = voc_fpath + "-relations.csv"
            print("Generating new relations file:", relations_fpath)
            self._relations_fpath = voc_fpath + "-relations.csv"
            self._relations = self._generate_relations(self.voc, self._relations_fpath)
        elif exists(relations_fpath):
            print("Loading relations file:", relations_fpath)
            self._relations_fpath = relations_fpath
            self._relations = read_csv(relations_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
            print("Loaded %d relations from: %s" % (len(self._relations), relations_fpath))
            hypos_voc = set(self._relations.hyponym.to_dict().values())
            hyper_voc = set(self._relations.hypernym.to_dict().values())
            self.voc = hypos_voc.union(hyper_voc)
            print("Loaded %d voc from relations" % len(self.voc))
        else:
            raise Exception("Error: cannot load relations or generate them. Specify either voc_fpath or relations_fpath.")


    def _str_in_str(self, substr, supstr):
        substr = str(substr).lower()
        supstr = str(supstr).lower()

        if len(substr) < 5: return 0
        
        index = supstr.find(substr)
        if index == -1:
            substr_l = lemmatize(substr) 
            index = supstr.find(substr_l)
            if index == -1:
                supstr_l = lemmatize(supstr)
                index = supstr_l.find(substr_l)
                if index == -1:
                    index = supstr_l.find(substr)
                    if index == -1: 
                        return 0, index

        return float(len(substr)) / float(len(supstr)), index
        
    def _generate_relations(self, voc, relations_fpath):
        with codecs.open(relations_fpath, "w", "utf-8") as out:
            print("relation_id\thyponym\thypernym\tcorrect", file=out)
            relation_id = 0
            for hypo in voc:
                for hyper in voc:
                    if hypo == hyper: continue
                    print("%d\t%s\t%s\t0" % (relation_id, hypo, hyper), file=out)
                    relation_id += 1  
        
        print("Generated %d relations out of %d words: %s" % (relation_id+1, len(voc), relations_fpath))
        return read_csv(relations_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
 
    def _load_voc(self, voc_fpath):
        if exists(voc_fpath):
            voc_df = read_csv(voc_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
            voc_name = fpath2filename(voc_fpath)

            voc = set()
            for i, row in voc_df.iterrows():
                if "term" in row: voc.add(row.term)
                elif "word" in row: voc.add(row.word)
            print("Loaded %d words vocabulary"  % len(voc)) 
            return voc
        else:
            print("Warning: vocabulary is not loaded. This means hypo2hyper features cannot be extracted.")
            return set()

    @property
    def isas(self):
        return self._isas

    @property
    def freq(self):
        """ Returns default freq dictionary. """

        if len(self._freqs) > 0:
            first_freq = list(self._freqs.keys())[0]
            return self._freqs[first_freq]

    @property
    def isa(self):
        """ Returns default isa dictionary. """

        if len(self._isas) > 0:
            first_isa = list(self._isas.keys())[0]
            return self._isas[first_isa]
    
    def fill_direct_isas_substrings_slow(self):
        for isa_name in self._isas:
            self._relations = self._fill_isas(self._isas[isa_name], self._relations, field_name_postfix="_" + isa_name + "_s", subphrases=True)
        self._relations = self._fill_average_isas(self._relations, field_name_postfix="_s")
        self._save_relations()

    def substract_feature(self, direct_field, reverse_field, result_field, normalize=True):
        result = np.zeros(len(self._relations))

        for i, row in self._relations.iterrows():
            if i != 0 and i % 100000 == 0: print(i)
            direct = row[direct_field]
            reverse = row[reverse_field]
            result[i] = direct - reverse
            if normalize: result[i] = result[i]/max(1, max(direct, reverse))

        self._relations[result_field] = Series(result, index=self._relations.index)
        self._save_relations()

    def fill_features_slow(self, relations_fpath):
        """ Extracts features for a given relations file. """

        tic = time()
        relations = read_csv(relations_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
        relations = self._fill_substrings(relations)
        relations = self._fill_frequencies(self.freq, relations)
        for isa_name in sorted(self._isas, reverse=True):
            print(isa_name)
            relations = self._fill_isas(self._isas[isa_name], relations, field_name_postfix="_" + isa_name, subphrases=False)
            relations = self._fill_isas(self._isas[isa_name], relations, field_name_postfix="_" + isa_name + "_s", subphrases=True)
            relations = self._fill_isas(self._isas[isa_name], relations, field_name_postfix="_" + isa_name + "_2", subphrases=False, proxy=True)
            relations = self._fill_degrees(self._isas[isa_name], relations, field_name_postfix="_" + isa_name)
        relations = self._fill_average_isas(relations)
        relations = self._fill_average_isas(relations, field_name_postfix="_s")

        relations.to_csv(relations_fpath, sep="\t", encoding="utf-8", float_format='%.0f', index=False)
        print("Updated relations:", relations_fpath)
        print("Features extracted in %d sec." % (time()-tic))

        return relations

    @property 
    def relations(self):
        return self._relations

    def _save_relations(self):
        self._relations.to_csv(self._relations_fpath, sep="\t", encoding="utf-8", float_format='%.5f', index=False)
        print("Relations updated:", self._relations_fpath)
 
    def hypo2hyper_ratio(self):
        self._relations["hypo2hyper_substract"] = self._relations["hypo2hyper"] - self._relations["hyper2hypo"] 
        self._save_relations()

    def fill_length(self):
        hypo_length = np.zeros(len(self._relations))
        hyper_length = np.zeros(len(self._relations))
        
        for i, row in self._relations.iterrows():
            try: 
                if i != 0 and i % 100000 == 0: print(i)
                hypo_length[i] = len(row.hyponym)
                hyper_length[i] = len(row.hypernym)
            except:
                print("Error:", row.hyponym, row.hypernym)
                print(format_exc())

        self._relations["hypo_length"] = Series(hypo_length, index=self._relations.index)
        self._relations["hyper_length"] = Series(hyper_length, index=self._relations.index)
        
        self._save_relations()
        
    def fill_substrings(self, must_have_space=True):
        self._relations = self._fill_substrings(self._relations, must_have_space)
        self._update_substrings()
        self._save_relations()
 
        debug_fpath = self._relations_fpath + "-substrings.csv"
        df = self._relations.copy() 
        df = df[(df.hyper_in_hypo != 0) | (df.hypo_in_hyper != 0)]
        df = df.sort_values(["hyponym", "hypo_in_hyper"], ascending=[1,0])
        df.to_csv(debug_fpath, sep="\t", encoding="utf-8", float_format='%.3f', index=False)
        print("Substrings:", debug_fpath)

    def _fill_substrings(self, relations, must_have_space):
        hyper_in_hypo = np.zeros(len(relations))
        hypo_in_hyper = np.zeros(len(relations))
        hyper_in_hypo_i = np.zeros(len(relations))

        for i, row in relations.iterrows():
            try: 
                if i != 0 and i % 100000 == 0: print(i)

                if (must_have_space and " " not in row.hypernym) or len(row.hyponym) < 5:
                    hypo_in_hyper[i] = 0 
                else:
                    hypo_in_hyper[i], _ = self._str_in_str(row.hyponym, row.hypernym)

                if (must_have_space and " " not in row.hyponym) or len(row.hypernym) < 5: hyper_in_hypo[i] = 0 
                else: hyper_in_hypo[i], _ = self._str_in_str(row.hypernym, row.hyponym)
                
                hyper_in_hypo_i[i] = 1. / hyper_in_hypo[i] if hyper_in_hypo[i] > 0 and hypo_in_hyper[i] == 0 else 0
            except:
                print("Error:", row.hyponym, row.hypernym)
                print(format_exc())

        relations["hyper_in_hypo"] = Series(hyper_in_hypo, index=relations.index)
        relations["hypo_in_hyper"] = Series(hypo_in_hyper, index=relations.index)
        relations["hyper_in_hypo_i"] = Series(hyper_in_hypo_i, index=relations.index)

        return relations
     
    def _update_substrings(self):
        if self._lang in ["en", "nl"]:
            self._update_supstr_in_substr_en_nl(self._relations, "hyponym", "hypernym", "hyper_in_hypo") 
            self._update_supstr_in_substr_en_nl(self._relations, "hypernym", "hyponym", "hypo_in_hyper") 
        else:
            self._update_supstr_in_substr_fr_it(self._relations, "hyponym", "hypernym", "hyper_in_hypo") 
            self._update_supstr_in_substr_fr_it(self._relations, "hypernym", "hyponym", "hypo_in_hyper") 
        self._update_hypo_in_hyper_i(self._relations)

    def _is_identical(self, str1, str2):
        str1 = re_dash.sub(" ", str1)
        str2 = re_dash.sub(" ", str2)
        return lemmatize(str1) == lemmatize(str2)

    def _update_supstr_in_substr_en_nl(self, relations, supstr_field, substr_field, res_field):
        df = relations.copy()
        df = df[df[res_field] > 0]

        for i, row in df.iterrows():
            supstr_tokens = row[supstr_field].split(" ")
            substr_tokens = row[substr_field].split(" ")
            if len(supstr_tokens) == 0 or len(substr_tokens) == 0:
                continue
            elif self._is_identical(row[supstr_field], row[substr_field]): 
                relations.loc[i, res_field] = 1.0 
                continue

            if len(set(supstr_tokens).intersection(self._stopwords)) > 0:
                supstr = " ".join(supstr_tokens[0:min(len(substr_tokens),len(supstr_tokens))])
                
                if self._str_in_str(row[substr_field], supstr)[0] > 0: 
                     substr_in_supstr = len(row[substr_field]) / float(len(row[supstr_field]))
                else:
                     substr_in_supstr = 0
            else:
                substr_in_supstr, subste_start_index = self._str_in_str(row[substr_field], row[supstr_field])
                if len(row[supstr_field]) > subste_start_index + len(row[substr_field]) + 3: substr_in_supstr = 0  # must be in the end
            if VERBOSE: print(i, substr_in_supstr, row[supstr_field], "-->", row[substr_field])

            relations.loc[i, res_field] = substr_in_supstr

    def _update_supstr_in_substr_fr_it(self, relations, supstr_field, substr_field, res_field):
        df = relations.copy()
        df = df[df[res_field] > 0]

        for i, row in df.iterrows():
            supstr_tokens = row[supstr_field].split(" ")
            substr_tokens = row[substr_field].split(" ")
            if len(supstr_tokens) == 0 or len(substr_tokens) == 0: continue

            substr_in_supstr, subste_start_index = self._str_in_str(row[substr_field], row[supstr_field])
            if subste_start_index != 0 and len(supstr_tokens) > 1: substr_in_supstr = 0  # must be in the beginning unless for single words

            if VERBOSE: print(i, substr_in_supstr, row[supstr_field], "-->", row[substr_field])

            relations.loc[i, res_field] = substr_in_supstr


    def _update_hypo_in_hyper_i(self, relations):
        relations["hyper_in_hypo_i"] = 0
        df = relations.copy()
        df = df[df["hyper_in_hypo"] > 0]
        df = df[df["hypo_in_hyper"] == 0]
        for i, row in df.iterrows():
            relations.loc[i, "hyper_in_hypo_i"] = 1. /  row["hyper_in_hypo"] 
        
    def fill_frequencies(self):
        relations = self._fill_frequencies(self.freq, self._relations)
        self._save_relations()

    def _fill_frequencies(self, freq, relations, field_name_postfix=""):
        hyponym_freq = np.zeros(len(relations))
        hypernym_freq = np.zeros(len(relations))

        for i, row in relations.iterrows():
            hyponym_freq[i] = freq.freq(row.hyponym)
            hypernym_freq[i] = freq.freq(row.hypernym)
        relations["hyponym_freq" + field_name_postfix] = Series(hyponym_freq, index=relations.index)
        relations["hypernym_freq" + field_name_postfix] = Series(hypernym_freq, index=relations.index)

        return relations

    def _proxy_path(self, hypo, hyper, isas, proxy_type="max", max_top=MAX_PROXY_ISAS): 
        """ proxy_type in 'max', 'mean', 'num' """
        
        path_weight = 0
        path = [] 
        hypo2proxy = {proxy: freq for proxy, freq in isas.all_hyper(hypo, max_top)}

        if hyper not in hypo2proxy:
            hypo2proxy.pop(hypo, None)
            proxy2hyper = {proxy: freq for proxy, freq in isas.all_hypo(hyper, max_top)}
            proxy2hyper.pop(hyper, None)
            hypo2hyper = {proxy: (hypo2proxy[proxy] + proxy2hyper[proxy])/2. 
                          for proxy in set(hypo2proxy.keys()).intersection(set(proxy2hyper.keys()))}
            hypo2hyper_s = sorted(list(hypo2hyper.items()), key=operator.itemgetter(1), reverse=True)
            hypo2hyper_s = [(proxy, weight) for proxy, weight in hypo2hyper_s if self.freq.freq(hypo) < self.freq.freq(proxy) and self.freq.freq(proxy) < self.freq.freq(hyper)] 
            # print proxy paths
            if VERBOSE:
                for proxy in hypo2hyper_s:
                    print("%.1f::: %s:%d --> %s:%d --> %s:%d" % (proxy[1], hypo, self.freq.freq(hypo), proxy[0], self.freq.freq(proxy[0]), hyper, self.freq.freq(hyper)))
            
            if len(hypo2hyper_s) > 0:
                if VERBOSE:
                    print("proxy type:", proxy_type)
                    for proxy, freq in hypo2hyper_s: print(hypo, "-->", proxy, ":", freq, "-->", hyper) 
                max_proxy = hypo2hyper_s[0][0]
                path = [(hypo, hypo2proxy[max_proxy]), max_proxy, (hyper, proxy2hyper[max_proxy])]
                
                if proxy_type == "max":
                    path_weight = hypo2hyper_s[0][1]
                elif proxy_type == "mean":
                    path_weight = mean(list(hypo2hyper.values()))
                elif proxy_type == "num":
                    path_weight = len(hypo2hyper_s)
                else:
                    path_weight = hypo2hyper_s[0][1]
        else:
            path = [hypo, hyper]
            path_weight = isas.data[hypo][hyper]
            
        return path, path_weight

    def fill_features(self):
        print("Number of relations:", len(self._relations))
        self.fill_frequencies()
        self.fill_substrings()
        self.fill_direct_isas(subphrases=False)
        self.fill_direct_isas(subphrases=True)
        self.fill_proxy_isas(max_top=10, proxy_type="max")
        self.fill_proxy_isas(max_top=20, proxy_type="mean")
        self.fill_degrees()

    def fill_super_features(self):
        self.fill_direct_isas()
        self.fill_substrings(must_have_space=False)
        self.hypo2hyper_ratio()
        
        direct_field = "hypernym_freq"
        reverse_field = "hyponym_freq"
        res_field = "freq_substract"
        normalize=True
        self.fill_frequencies()
        self.substract_feature(direct_field, reverse_field, res_field, normalize)

        direct_field = "hyper_in_weight_en_ps"
        reverse_field = "hypo_in_weight_en_ps"
        res_field = "in_weight_substract"
        normalize=True
        self.fill_degrees()
        self.substract_feature(direct_field, reverse_field, res_field, normalize)

        direct_field = "hyper_length"
        reverse_field = "hypo_length"
        res_field = "length_substract"
        normalize=True
        self.fill_length()
        self.substract_feature(direct_field, reverse_field, res_field, normalize)

        direct_field = "hypo2hyper_s"
        reverse_field = "hyper2hypo_s"
        res_field = "hypo2hyper_s_substract"
        normalize=False
        self.fill_direct_isas(subphrases=True)
        self.substract_feature(direct_field, reverse_field, res_field, normalize)

        direct_field = "hypo2hyper_max2"
        reverse_field = "hyper2hypo_max2"
        res_field = "hypo2hyper_max2_substract"
        normalize=False
        self.fill_proxy_isas(max_top=10, proxy_type="max")
        self.substract_feature(direct_field, reverse_field, res_field, normalize)

    def fill_degrees(self):
        for isa_name in self._isas:
            print(isa_name)
            p = "_"+isa_name
            self._relations = self._fill_in_degrees(self._isas[isa_name], self._relations, field_name_postfix=p)
            self._relations = self._fill_out_degrees(self._isas[isa_name], self._relations, field_name_postfix=p)
        self._save_relations() 
 
    def fill_proxy_isas(self, max_top=MAX_PROXY_ISAS, proxy_type="max"):

        # calculate proxy features for separate models and their average
        p = "_" + proxy_type + "2"  
        self._relations["hypo2hyper" + p] = Series(np.zeros(len(self._relations)), index=self._relations.index)
        self._relations["hyper2hypo" + p] = Series(np.zeros(len(self._relations)), index=self._relations.index)
        isas_num = float(len(self._isas))
        for isa_name in self._isas:
            print("Calculating proxy isas from:", isa_name)
            postfix = "_" + isa_name + p
            self._relations = self._fill_isas(self._isas[isa_name], self._relations, 
                field_name_postfix=postfix,max_top=max_top, 
                subphrases=False, proxy_type=proxy_type) 
            
            # accumulate average
            isa_field_name = "isa_freq" + postfix
            max_hypo2hyper = float(max(self._relations[isa_field_name].max(), 1))
            self._relations["hypo2hyper" + p] += self._relations[isa_field_name] / max_hypo2hyper / isas_num
      
            ais_field_name = "ais_freq" + postfix
            max_hyper2hypo = float(max(self._relations[ais_field_name].max(), 1))
            self._relations["hyper2hypo" + p] += self._relations[ais_field_name] / max_hyper2hypo / isas_num
        self._save_relations() 
        
        # save extra results
        relations_fpath = self._relations_fpath + postfix + "-hypo2hyper.csv"
        df = self._relations.copy()
        df = df[(df["hypo2hyper" + p] != 0) | (df["hypo2hyper" + p] != 0)]
        df = df.sort_values(["hyponym", "hypo2hyper" + p], ascending=[1,0])
        df.to_csv(relations_fpath, sep="\t", encoding="utf-8", float_format='%.5f', index=False)
        print("Proxy relations:", relations_fpath)
        
    def fill_direct_isas(self, subphrases=False):
        # get direct hypernyms of different isas: model_name -> (hypo, hyper) -> weight
        hypo2hyper_freq = defaultdict(dict)   # raw frequency
        hypo2hyper_inorm = defaultdict(dict)  # in-voc norm: divide by max invoc frequency
        hypo2hyper_anorm = defaultdict(dict)  # absolute norm: divide by max frequency per word

        for isa_name in self._isas:
            print(isa_name, len(self._isas[isa_name].data))
            for hypo in self.voc:
                # find hypernyms
                hypers_list = self._isas[isa_name].all_hyper(hypo)
                hypers_dict = {hyper: freq for hyper, freq in hypers_list}
                invoc_hypers_dict = {w: hypers_dict[w] for w in set(hypers_dict.keys()).intersection(self.voc)}
                invoc_hypers_dict.pop(hypo, None)
                invoc_hypers_list = sorted(list(invoc_hypers_dict.items()), key=operator.itemgetter(1), reverse=True)
                
                if VERBOSE:
                    if len(invoc_hypers_list) > 0:
                        print(hypo, len(hypers_dict), len(invoc_hypers_list), ", ".join(w + ":" + str(freq) for w, freq in invoc_hypers_list))
                    # print len(invoc_hypers_list),
                
                # find hypernyms of subphrases
                if len(invoc_hypers_list) == 0 and subphrases:
                    for hypo_subphrase in self._subphrases(hypo):
                        hypers_list = self._isas[isa_name].all_hyper(hypo_subphrase)
                        hypers_dict = {hyper: freq for hyper, freq in hypers_list}
                        invoc_hypers_dict = {w: hypers_dict[w] for w in set(hypers_dict.keys()).intersection(self.voc)}
                        invoc_hypers_dict.pop(hypo_subphrase, None)
                        invoc_hypers_list = sorted(list(invoc_hypers_dict.items()), key=operator.itemgetter(1), reverse=True)
                        if (invoc_hypers_list) > 0: break
                    if len(invoc_hypers_list) == 0:
                        continue
                    elif VERBOSE:
                        print(hypo, "-->", hypo_subphrase, ":", invoc_hypers_list)
                elif len(invoc_hypers_list) == 0:
                    continue
                
                # normalize
                max_freq = float(hypers_list[0][1])
                invoc_max_freq = float(invoc_hypers_list[0][1])
                for hyper, freq in invoc_hypers_list:
                    hypo2hyper_freq[isa_name][(hypo, hyper)] = freq
                    hypo2hyper_anorm[isa_name][(hypo, hyper)] = freq/max_freq
                    hypo2hyper_inorm[isa_name][(hypo, hyper)] = freq/invoc_max_freq
        # average: (hypo, hyper) -> weight
        hypo2hyper_iavg = self._average(hypo2hyper_inorm)
        hypo2hyper_aavg = self._average(hypo2hyper_anorm)

        # initialize arrays
        hyper2hypo_iavg_arr = np.zeros(len(self._relations))
        hypo2hyper_iavg_arr = np.zeros(len(self._relations))
        hypo2hyper_iavg2_arr = np.zeros(len(self._relations))
        hyper2hypo_iavg2_arr = np.zeros(len(self._relations))
        hypo2hyper_aavg_arr = np.zeros(len(self._relations))
        hyper2hypo_aavg_arr = np.zeros(len(self._relations))
        hypo2hyper_arr = {}
        hyper2hypo_arr = {}
        for isa_name in hypo2hyper_inorm:
            hypo2hyper_arr[isa_name] = np.zeros(len(self._relations))
            hyper2hypo_arr[isa_name] = np.zeros(len(self._relations))

        # fill the arrays
        for i, row in self._relations.iterrows():
            if i != 0 and i % 100000 == 0: print(i)
            hypo2hyper_iavg_arr[i] = hypo2hyper_iavg.pop((row.hyponym, row.hypernym), 0)
            hyper2hypo_iavg_arr[i] = hypo2hyper_iavg.pop((row.hypernym, row.hyponym), 0)
            hypo2hyper_aavg_arr[i] = hypo2hyper_aavg.pop((row.hyponym, row.hypernym), 0)
            hyper2hypo_aavg_arr[i] = hypo2hyper_aavg.pop((row.hypernym, row.hyponym), 0)
            for isa_name in hypo2hyper_inorm:
                hypo2hyper_arr[isa_name][i] = hypo2hyper_freq[isa_name].pop((row.hyponym, row.hypernym), 0)
                hyper2hypo_arr[isa_name][i] = hypo2hyper_freq[isa_name].pop((row.hypernym, row.hyponym), 0)

        # insert arrays as columns
        s = "_s" if subphrases else ""
        for isa_name in hypo2hyper_inorm:
            col = "hypo2hyper_" + isa_name + s
            self._relations[col] = Series(hypo2hyper_arr[isa_name], index=self._relations.index)
            hypo2hyper_iavg2_arr += self._relations[col] / self._relations[col].max()

            col = "hyper2hypo_" + isa_name + s
            self._relations[col] = Series(hyper2hypo_arr[isa_name], index=self._relations.index)
            hyper2hypo_iavg2_arr += self._relations[col] / self._relations[col].max()

        self._relations["hypo2hyper" + s] = Series(hypo2hyper_iavg_arr, index=self._relations.index)
        self._relations["hyper2hypo" + s] = Series(hyper2hypo_iavg_arr, index=self._relations.index)
        self._relations["hypo2hyper2" + s] = Series(hypo2hyper_iavg2_arr, index=self._relations.index)
        self._relations["hyper2hypo2" + s] = Series(hyper2hypo_iavg2_arr, index=self._relations.index)
        self._relations["hypo2hyper3" + s] = Series(hypo2hyper_aavg_arr, index=self._relations.index)
        self._relations["hyper2hypo3" + s] = Series(hyper2hypo_aavg_arr, index=self._relations.index)
        self._save_relations()        
        
        # debug info
        debug_fpath = self._relations_fpath + "-direct-hypo2hyper" + s + ".csv"
        tmp_fpath = debug_fpath + ".tmp"
        with codecs.open(tmp_fpath, "w", "utf-8") as out:
            print("hyponym\thypernym\tfreq", file=out)
            for hypo, hyper in hypo2hyper_iavg: print("%s\t%s\t%.3f" % (hypo, hyper, hypo2hyper_iavg[(hypo, hyper)]), file=out)
        df = read_csv(tmp_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
        df = df.sort_values(["hyponym","freq"], ascending=[1,0])
        df.to_csv(debug_fpath, sep="\t", encoding="utf-8", float_format='%.3f', index=False)
        try_remove(tmp_fpath)
        print("Direct hypernyms:", debug_fpath)

    def _average(self, hypo2hyper_inorm):
        hypo2hyper_iavg = {}
        num_models = float(len(hypo2hyper_inorm))
        for isa_name in hypo2hyper_inorm:
            for hypo_hyper in hypo2hyper_inorm[isa_name]:
                if hypo_hyper in hypo2hyper_iavg:
                    hypo2hyper_iavg[hypo_hyper] += hypo2hyper_inorm[isa_name][hypo_hyper] / num_models
                else:
                    hypo2hyper_iavg[hypo_hyper] = hypo2hyper_inorm[isa_name][hypo_hyper] / num_models

        return hypo2hyper_iavg

    def _subphrases(self, term):
        tokens = split('\W+', term)
        subphrases = []
        
        for i in range(len(tokens)):
            if i == 0: continue
            if self._lang in ["en", "nl"]: term_cut = " ".join(tokens[i:])
            elif  self._lang in ["fr", "it"]: term_cut = " ".join(tokens[:-i])
            else: term_cut = " ".join(tokens[i:])
            subphrases.append(term_cut)
        
        return subphrases

    def _fill_isas(self, isas, relations, field_name_postfix="", subphrases=False, proxy_type="", max_top=MAX_PROXY_ISAS):
        """ proxy_type in 'max', 'mean', 'num' """

        isa_freq = np.zeros(len(relations))
        ais_freq = np.zeros(len(relations))

        for i, row in relations.iterrows():
            if i != 0 and i % 10000 == 0: print(i, end=' ')
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

            if proxy_type != "" and isa_freq[i] == 0 and ais_freq[i] == 0:
                _, isa_freq[i] = self._proxy_path(row.hyponym, row.hypernym, isas, proxy_type=proxy_type, max_top=max_top)
                _, ais_freq[i] = self._proxy_path(row.hypernym, row.hyponym, isas, proxy_type=proxy_type, max_top=max_top)

        relations["isa_freq" + field_name_postfix] = Series(isa_freq, index=relations.index)
        relations["ais_freq" + field_name_postfix] = Series(ais_freq, index=relations.index)

        print("")
        return relations

    def _proxy_path_mean(self, hypo, hyper, isas):
        direct_isas = isas.all_hyper(hypo, MAX_PROXY_ISAS)

        paths = []
        path_weights = [0.]
        if hyper not in direct_isas:
            for proxy, _ in direct_isas:
                proxy_isas_freqs = isas.all_hyper(proxy, MAX_PROXY_ISAS)
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

    def _proxy_path_max(self, hypo, hyper, isas):
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
        return best_path, best_path_weight

    def _fill_average_isas(self, relations, field_name_postfix=""):
        p = field_name_postfix

        sum_isas = Series(np.zeros(len(relations)), index=relations.index)
        sum_aiss = Series(np.zeros(len(relations)), index=relations.index)
        for isa_name in self._isas:
            max_freq = max(relations["isa_freq_" + isa_name + p].max(), relations["ais_freq_" + isa_name + p].max())
            print(max_freq)
            sum_isas += relations["isa_freq_" + isa_name + p] / max_freq
            sum_aiss += relations["ais_freq_" + isa_name + p] / max_freq

        relations["isa_freq" + p] = sum_isas / float(len(self._isas))
        relations["ais_freq" + p] = sum_aiss / float(len(self._isas))

        return relations

    def _fill_in_degrees(self, isas, relations, field_name_postfix=""):
        hypo_in_num = np.zeros(len(relations))
        hyper_in_num = np.zeros(len(relations))
        hypo_in_weight = np.zeros(len(relations))
        hyper_in_weight = np.zeros(len(relations))

        for i, row in relations.iterrows():
            if i % 1000 == 0 and i != 0: print(i, end=' ')
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

    def _fill_out_degrees(self, isas, relations, field_name_postfix=""):
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

