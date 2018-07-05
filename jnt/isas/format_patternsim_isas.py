import argparse
import codecs
from pandas import read_csv
from collections import defaultdict
from traceback import format_exc
import operator
import re
from os.path import splitext


DEBUG = False
CHUNK_SIZE=1000000
re_comma = re.compile(r"^([^,]*),.*", re.U|re.I)


def clean_patternsim_term(term):
    cterm = str(term)
    cterm = re_comma.sub(r"\1", cterm)
    return cterm

def patternsim2isas(patternsim_fpath, output_fpath):
    patternsim_df = read_csv(patternsim_fpath, encoding='utf-8', delimiter=";", error_bad_lines=False, low_memory=True)
    print("Loaded %d pairs" % len(patternsim_df))
    isas = defaultdict(dict)
    
    for i, row in patternsim_df.iterrows():
        try:
            if i % 100000 == 0: print(i)
            
            word1 = clean_patternsim_term(row.form)
            word2 = clean_patternsim_term(row.related)
            word1_isa_word2 = int(row.hypo)
            word2_isa_word1 = int(row.hyper)
            
            if (word1 not in isas or word2 not in isas[word1]) and word1_isa_word2 > 0:
                isas[word1][word2] = word1_isa_word2
            elif word1_isa_word2 > 0:
                isas[word1][word2] += word1_isa_word2

            if (word2 not in isas or word1 not in isas[word2]) and word2_isa_word1 > 0:
                isas[word2][word1] = word2_isa_word1
            elif word2_isa_word1 > 0:
                isas[word2][word1] += word2_isa_word1
        except:
            pass
            #print "Bad row:", row
            print(format_exc())

    with codecs.open(output_fpath, "w", "utf-8") as out:
        print("hyponym\thypernym\tfreq", file=out)
        for hypo in isas:
            for hyper, freq in sorted(list(isas[hypo].items()), key=operator.itemgetter(1), reverse=True):
                if isas[hypo][hyper] <= 0:
                    print("Skipping '%s' --(%d)--> '%s'" % (hypo, isas[hypo][hyper], hyper))
                    continue
                print("%s\t%s\t%d" % (hypo, hyper, freq), file=out)

    print("Output:", output_fpath)



def patternsim2isas_hh(hh_fpath, output_fpath):
    """ Transforms file 'word1<TAB>word2<TAB>word1_isa_word2<TAB>word2_isa_word1' to 'hyponym<TAB>hypernym<TAB>freq'."""
    hh_df = read_csv(hh_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False, low_memory=False)
    isas = defaultdict(dict)

    for i, row in hh_df.iterrows():
        try:
            #if i > 100000: break
            if i % 100000 == 0: print(i)
            word1 = clean_patternsim_term(row.word1)
            word2 = clean_patternsim_term(row.word2)
            word1_isa_word2 = int(row.word1_isa_word2)
            word2_isa_word1 = int(row.word2_isa_word1)
            if word1 not in isas or word2 not in isas[word1]:
                isas[word1][word2] = word1_isa_word2
            else:
                isas[word1][word2] += word1_isa_word2

            if word2 not in isas or word1 not in isas[word2]:
                isas[word2][word1] = word2_isa_word1
            else:
                isas[word2][word1] += word2_isa_word1
        except:
            print("Bad row:", row)
            print(format_exc())

    with codecs.open(output_fpath, "w", "utf-8") as out:
        print("hyponym\thypernym\tfreq", file=out)
        for hypo in isas:

            for hyper, freq in sorted(list(isas[hypo].items()), key=operator.itemgetter(1), reverse=True):
                if isas[hypo][hyper] <= 0: continue
                print("%s\t%s\t%d" % (hypo, hyper, freq), file=out)

    print("Output:", output_fpath)


def main():
    parser = argparse.ArgumentParser(description="Transforms PatternSim output pairs.csv to a CSV file "
                                                 "'hyponym<TAB>hypernym<TAB>freq' with a header.")
    parser.add_argument('inp', help='Path to an input file.')
    parser.add_argument('-o', help='Output file. Default -- next to input file.', default="")
    args = parser.parse_args()

    output_fpath = splitext(args.inp)[0] + "-isas.csv" if args.o == "" else args.o
    print("Input: ", args.inp)
    print("Output: ", output_fpath)
    patternsim2isas(args.inp, output_fpath)

if __name__ == '__main__':
    main()


