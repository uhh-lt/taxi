import argparse
from jnt.isas.taxo import TaxonomyFeatures, TaxonomyResources
from jnt.isas.predictors import TaxonomyPredictor
from glob import glob
from traceback import format_exc
from os.path import join



RES_DIR = "./resources"
CLASSIFIERS_DIR = join(RES_DIR,"models/release/2features-new/*")


def load_res(language, mode, test_en=False):
    if language == "en":
        if mode == "simple": freq_fpaths=[""]
        else: freq_fpaths = [join(RES_DIR,"en_freq-59g-mwe62m.csv.gz")]

        if test_en:
            isa_common_fpaths = [join(RES_DIR,"en_ps.csv.gz")]
        else:
            isa_common_fpaths = [
                   join(RES_DIR,"en_ma.csv.gz"),
                   join(RES_DIR,"en_pm.csv.gz"),
                   join(RES_DIR,"en_ps.csv.gz"),
                   join(RES_DIR,"en_cc.csv.gz"),
                   join(RES_DIR,"en_ps59g.csv.gz")]

        isa_domain_fpaths = {
            "food": [join(RES_DIR,"en_food.csv.gz")],
            "science": [join(RES_DIR,"en_science.csv.gz")],
            "environment": [join(RES_DIR,"en_environment.csv.gz")]}

    elif language == "fr":
        freq_fpaths=[""]
        isa_common_fpaths = [join(RES_DIR,"fr.csv.gz")]
        isa_domain_fpaths = {
            "food": [join(RES_DIR,"fr_food.csv.gz")],
            "science": [join(RES_DIR,"fr_science.csv.gz")],
            "environment": [join(RES_DIR,"fr_environment.csv.gz")]}

    elif language == "nl":
        freq_fpaths=[""]
        isa_common_fpaths = [join(RES_DIR,"nl.csv.gz")]
        isa_domain_fpaths = {
            "food": [join(RES_DIR,"nl_food.csv.gz")],
            "science": [join(RES_DIR,"nl_science.csv.gz")],
            "environment": [join(RES_DIR,"nl_environment.csv.gz")]}

    elif language == "it":
        freq_fpaths=[""]
        isa_common_fpaths = [join(RES_DIR,"it.csv.gz")]
        isa_domain_fpaths = {
            "food": [join(RES_DIR,"it_food.csv.gz")],
            "science": [join(RES_DIR,"it_science.csv.gz")],
            "environment": [join(RES_DIR,"it_environment.csv.gz")]}

    taxo_res_domain = {}
    for domain in isa_domain_fpaths:
        taxo_res_domain[domain] = TaxonomyResources(freq_fpaths=[], isa_fpaths=isa_domain_fpaths[domain])
    taxo_res_common = TaxonomyResources(freq_fpaths=freq_fpaths, isa_fpaths=isa_common_fpaths)

    return taxo_res_common, taxo_res_domain


def get_taxo_res_domain_voc(taxo_res_domain, voc_fpath):
    for domain in taxo_res_domain.keys():
        if domain in voc_fpath:
            print(voc_fpath, "is", domain)
            return taxo_res_domain[domain]

    print("Warning: domain not found for", voc_fpath)
    return TaxonomyResources()


def combine_taxo_res(taxo_res1, taxo_res2):
    taxo_res12 = TaxonomyResources()

    taxo_res12._isas = taxo_res1._isas.copy()
    taxo_res12._isas.update(taxo_res2._isas)

    taxo_res12._freqs = taxo_res1._freqs.copy()
    taxo_res12._freqs.update(taxo_res2._freqs)

    return taxo_res12


def evaluate_on_trial_taxo():
    relations_fpath = join(RES_DIR,"relations.csv")  # assuming features "hyper_in_hypo_i" and "hypo2hyper_substract"
    taxo_fpath = relations_fpath + "-taxo.csv"
    print("Relations:", relations_fpath)
    print("Unpruned taxonomy:", taxo_fpath)

    taxo_features = TaxonomyFeatures(TaxonomyResources(), relations_fpath=relations_fpath, lang="en")
    taxo_predict = TaxonomyPredictor(taxo_features)
    taxo_predict.predict_by_global_threshold(threshold=0, field="hypo2hyper_substract", or_correct_predict=False)
    taxo_predict.predict_by_global_threshold(threshold=0, field="hyper_in_hypo_i", or_correct_predict=True)
    taxo_predict.save(taxo_fpath)
    taxo_predict.evaluate(field="correct_predict")

    for max_knn in [1, 2, 3, 5]:
        taxo_knn_fpath = relations_fpath + "-taxo-knn" + str(max_knn) + ".csv"
        taxo_predict.predict_by_local_threshold(threshold=0, max_knn=max_knn, field="hypo2hyper_substract", or_correct_predict=False)
        taxo_predict.predict_by_global_threshold(threshold=0, field="hyper_in_hypo_i", or_correct_predict=True)
        taxo_predict.save(taxo_knn_fpath)
        taxo_predict.evaluate(field="correct_predict")


def extract_semeval_taxo(input_voc_pattern, language, mode, classifiers_pattern, test_en):
    #Laedt alle Datensaetze(auch alle Domaenen, aus vocabularies)
    taxo_res_common, taxo_res_domain = load_res(language, mode, test_en)

    for voc_fpath in sorted(glob(input_voc_pattern)):
        for space in [False, True]:
            s = "-space" if space else ""
            relations_fpath = voc_fpath + s + "-relations.csv"
            taxo_fpath = relations_fpath + "-taxo.csv"
            print("\n", voc_fpath, "\n", "="*50)
            print("Relations:", relations_fpath)
            print("Unpruned taxonomy:", taxo_fpath)

            #Laedt domain-datenset und kombiniert sie mit dem allgemeinen Datenset
            taxo_res_domain_voc = get_taxo_res_domain_voc(taxo_res_domain, voc_fpath)
            taxo_res_voc = combine_taxo_res(taxo_res_common, taxo_res_domain_voc)
            taxo_features = TaxonomyFeatures(taxo_res_voc, voc_fpath, lang=language)

            if mode == "simple":
                taxo_features.fill_direct_isas()
                taxo_features.fill_substrings(must_have_space=space)
                taxo_features.hypo2hyper_ratio()
                taxo_predict = TaxonomyPredictor(taxo_features)
                taxo_predict.predict_by_global_threshold(threshold=0, field="hypo2hyper_substract", or_correct_predict=False)
                taxo_predict.predict_by_global_threshold(threshold=0, field="hyper_in_hypo_i", or_correct_predict=True)
                taxo_predict.save(taxo_fpath)

                for max_knn in [1, 2, 3, 5]:
                    #hypo2hyper fuer pattern
                    #hyperinhypoi feur substring
                    taxo_knn_fpath = relations_fpath + "-taxo-knn" + str(max_knn) + ".csv"
                    taxo_predict.predict_by_local_threshold(threshold=0, max_knn=max_knn, field="hypo2hyper_substract", or_correct_predict=False)
                    taxo_predict.predict_by_global_threshold(threshold=0, field="hyper_in_hypo_i", or_correct_predict=True)
                    taxo_predict.save(taxo_knn_fpath)

            elif mode == "super":
                taxo_features.fill_super_features()

                for classifier_dir in glob(classifiers_pattern):
                    try:
                        print("Predicting with:", classifier_dir)
                        taxo_predict = TaxonomyPredictor(taxo_features)
                        method = taxo_predict.predict_by_classifier(classifier_dir)
                        taxo_predict.save(taxo_fpath + "-" + method + ".csv")
                        taxo_predict.save(taxo_fpath + "-" + method + "-conf.csv", conf=True)
                    except:
                        print(format_exc())


def main():
    parser = argparse.ArgumentParser(description="Apply classifiers to the trial data.")
    parser.add_argument('input', help='Input vocabulary pattern e.g. "/home/en/*_en.csv"')
    parser.add_argument('language', type=str, default='en', choices=['en', 'fr', 'nl', 'it'], help='Path to an input file.')
    parser.add_argument('mode', type=str, default='simple', choices=['simple', 'super'], help="Mode of the taxonomy induction system. Use 'simple' for the unsupervised method, 'super' for supervised method and 'test' for a quick test.")
    parser.add_argument('--test', action='store_true', help="Load only few resouses, but do it quickly (works only for English).")
    parser.add_argument('-c', help='Path to the classifier or a pattern to the classifiers e.g. "/home/*".', default=CLASSIFIERS_DIR)
    args = parser.parse_args()

    print("Input: ", args.input)
    print("Language: ", args.language)
    print("Mode: ", args.mode)
    print("Classifiers: ", args.c)
    print("Test model: ", args.test)

    if args.mode in ["simple", "super"]:
        extract_semeval_taxo(args.input, args.language, args.mode, args.c, args.test)
    else:
        evaluate_on_trial_taxo()

if __name__ == '__main__':
    main()
