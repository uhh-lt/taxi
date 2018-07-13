from jnt.isas.taxo import TaxonomyFeatures, TaxonomyResources
from jnt.common import ensure_dir
from jnt.isas.supervised import SuperTaxi, METHODS
from traceback import format_exc
from os.path import join
import itertools
import argparse


RES_DIR = "./resources"
TEST_GRIDSEARCH = True


train_relations_fpath = join(RES_DIR,"relations.csv")
isa_fpaths = [join(RES_DIR,"en_ma.csv.gz")]
freq_fpaths =[""]


def findsubsets(S,m):
    return set(itertools.combinations(S, m))

def run(output_dir, feature_num, mode):

    feature_num = int(feature_num)
    taxo_res = TaxonomyResources(freq_fpaths, isa_fpaths)
    taxo_features = TaxonomyFeatures(taxo_res, relations_fpath=train_relations_fpath)

    ensure_dir(output_dir)
    features = ["hyper_in_hypo_i","hypo2hyper_substract", "freq_substract", "in_weight_substract", "length_substract",
            "hypo2hyper_s_substract","hypo2hyper_max2_substract"]
    features = features[:feature_num]

    if mode == "gridsearch":
        #  grid search is only supported for SVC
        method = "SVC"
        hc = SuperTaxi(join(output_dir, "SVC-grid-search"), method="SVC", features=features, overwrite=True)
        clf = hc.grid_search_svc(taxo_features.relations, test=TEST_GRIDSEARCH)
        return

    for method in METHODS:
        try:
            classifier_dir = join(output_dir, method)
            print("\n", method.upper(), "\n", "="*50)
            hc = SuperTaxi(classifier_dir, method=method, features=features, overwrite=True)
            if mode == "train":
                clf = hc.train(taxo_features.relations)
                hc._print_clf_info()
            elif mode == "cv":
                hc.crossval(taxo_features.relations)
            else:
                print("Error: unrecognised mode %s" % mode)
        except:
            print(format_exc())


def main():
    parser = argparse.ArgumentParser(description="Apply classifiers to the trial data.")
    parser.add_argument('output_dir', help="Output directory where classifiers will be saved.")
    parser.add_argument('feature_num', help='Number of features')
    parser.add_argument('mode', type=str, default='simple', choices=['train','cv','gridsearch'], help="Mode of the system.")
    args = parser.parse_args()

    print "Output directory: ", args.output_dir
    print "Features number: ", args.feature_num
    print "Mode: ", args.mode
    run(args.output_dir, args.feature_num, args.mode)

if __name__ == '__main__':
    main()
