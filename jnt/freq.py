from pandas import read_csv
import _pickle as pickle
from traceback import format_exc

from .common import exists, preprocess_pandas_csv
from .common import try_remove


DEFAULT_FREQ = 1


def load_freq(freq_fpath, min_freq=1, preprocess=True, sep='\t', strip_pos=True, use_pickle=True):
    f = FreqDictionary(freq_fpath, min_freq=min_freq, preprocess=preprocess, sep=sep, strip_pos=strip_pos, use_pickle=use_pickle)
    return f.data


class FreqDictionary(object):
    def __init__(self, freq_fpath, min_freq=1, preprocess=True, sep='\t', strip_pos=True, use_pickle=True):
        """ Reads a word frequency list in CSV format "word<TAB>freq" """

        if not exists(freq_fpath):
            self._freq = {}
            return

        pkl_fpath = freq_fpath + ".pkl"
        if use_pickle and exists(pkl_fpath):
            voc = pickle.load(open(pkl_fpath, "rb"))
        else:
            # load words to datafame
            if preprocess:
                freq_cln_fpath = freq_fpath + "-cln"
                preprocess_pandas_csv(freq_fpath, freq_cln_fpath)
                word_df = read_csv(freq_cln_fpath, sep, encoding='utf-8', error_bad_lines=False)
                try_remove(freq_cln_fpath)
            else:
                word_df = read_csv(freq_fpath, sep, encoding='utf-8', error_bad_lines=False)

            # load from dataframe to dictionary
            word_df = word_df.drop(word_df[word_df["freq"] < min_freq].index)
            if strip_pos:
                voc = {}
                for i, row in word_df.iterrows():
                    try:
                        word = str(row["word"]).split("#")[0]
                        freq = int(row["freq"])
                        if word not in voc or voc[word] < freq: voc[word] = freq
                    except:
                        print("Bad row:", row)
                        print(format_exc())
            else:
                voc = { row["word"]: row["freq"] for i, row in word_df.iterrows() }

            print("dictionary is loaded:", len(voc))

            if use_pickle:
                pickle.dump(voc, open(pkl_fpath, "wb"))
                print("Pickled voc:", pkl_fpath)

        print("Loaded %d words from: %s" % (len(voc), pkl_fpath if pkl_fpath else freq_fpath))

        self._freq = voc


    @property
    def data(self):
        return self._freq

    def freq(self, word):
        """ Returns frequency of the word or 1 """

        if word in self._freq: return self._freq[word]
        else: return DEFAULT_FREQ
