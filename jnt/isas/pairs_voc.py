import argparse
import codecs
from pandas import read_csv
from traceback import format_exc
from os.path import splitext
from jnt.common import load_voc


def filter_by_voc(hh_fpath, voc_fpath, output_fpath, both_in_voc=False):
    with codecs.open(output_fpath, "w", "utf-8") as out:
        print("hyponym\thypernym\tfreq", file=out)
        voc = load_voc(voc_fpath, preprocess=True, sep='\t', use_pickle=True, silent=False)

        hh_df = read_csv(hh_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False, low_memory=False)

        for i, row in hh_df.iterrows():
            try:
                if i % 100000 == 0: print(i)
                if both_in_voc:
                    if row.hyponym in voc and row.hypernym in voc: print("%s\t%s\t%d" % (row.hyponym, row.hypernym, row.freq), file=out)
                else:
                    if row.hyponym in voc or row.hypernym in voc: print("%s\t%s\t%d" % (row.hyponym, row.hypernym, row.freq), file=out)
            except:
                print("Bad row:", row)
                print(format_exc())

        print("Output:", output_fpath)


def main():
    parser = argparse.ArgumentParser(description="Filters file 'hyponym<TAB>hypernym<TAB>freq' with header so that it "
                                                 "contains only words from the input vocabulary.")
    parser.add_argument('inp', help="Path to an input file.")
    parser.add_argument('voc', help="Path to a voc file in the format 'word' with header.")
    parser.add_argument('--both_in_voc', action='store_true', help='Both hyper and hypo are in voc, otherwise hyper or hypo'
                                                                   'should be in the input vocabulary. Default -- false')
    parser.add_argument('-o', help='Output file. Default -- next to input file.', default="")
    args = parser.parse_args()

    output_fpath = splitext(args.inp)[0] + "-isas.csv" if args.o == "" else args.o
    print("Input: ", args.inp)
    print("Voc: ", args.voc)
    print("Both in voc: ", args.both_in_voc)
    print("Output: ", output_fpath)
    filter_by_voc(args.inp, args.voc, output_fpath, both_in_voc=args.both_in_voc)

if __name__ == '__main__':
    main()


