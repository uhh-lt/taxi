import codecs
from pandas import read_csv
import re
from traceback import format_exc


DEBUG = False
CHUNK_SIZE=1000000

re_opening_brace = re.compile(r"^([^(]*)\(.*", re.U|re.I)
re_closing_brace = re.compile(r"^([^)]*)\).*", re.U|re.I)


def clean_term(term):
    cterm = str(term)
    cterm = re_opening_brace.sub(r"\1", cterm)
    cterm = re_closing_brace.sub(r"\1", cterm)
    cterm = cterm.strip()
    return cterm


def skip_term(term):
    if  "<" in term or ">" in term or "href=" in term or "_top" in term:
        return True
    else:
        return False


def clean_isas(cc_fpath, output_fpath, filtered_fpath):
    """ Cleans an isa file 'hyponym<TAB>hypernym<TAB>freq'. """

    with codecs.open(output_fpath, "w", encoding="utf-8") as output, codecs.open(filtered_fpath, "w", encoding="utf-8") as filtered:
        reader = read_csv(cc_fpath, encoding="utf-8", delimiter="\t", error_bad_lines=False,
            iterator=True, chunksize=CHUNK_SIZE, doublequote=False, quotechar="\u0000")

        num = 0
        selected_num = 0
        for i, chunk in enumerate(reader):
            # print header
            if i == 0:
                for field in chunk.columns.values:
                    output.write(field + "\t")
                output.write("\n")
            chunk.fillna('')

            # print rows
            for j, row in chunk.iterrows():
                try:
                    if num % 100000 == 0: print(num)
                    num += 1

                    hyper = clean_term(row.hypernym)
                    hypo = clean_term(row.hyponym)
                    freq = int(row.freq)
                    if skip_term(hyper) or skip_term(hypo) and hyper != "" and hypo != "" :
                        print("%s\t%s\t%d" % (row.hypernym, row.hyponym, row.freq), file=filtered)
                        continue

                    print("%s\t%s\t%d" % (hyper, hypo, freq), file=output)
                    selected_num += 1
                except:
                    print("Bad row:", row)
                    print(format_exc())

    print("Input # isas:", num)
    print("Output # of isas:", selected_num)
    print("Filtered:", filtered_fpath)
    print("Output:", output_fpath)


cc_fpath = "/home/panchenko/joint/data/isas/cc/cc.csv"
output_fpath = cc_fpath + "-out.csv"
filtered_fpath = cc_fpath + "-filtered.csv"

clean_isas(cc_fpath, output_fpath, filtered_fpath)