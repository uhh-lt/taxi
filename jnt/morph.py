from treetagger import TreeTagger
from spacy.en import English
from stop_words import get_stop_words
from pandas import read_csv
from os.path import join
from jnt.common import preprocess_pandas_csv, get_data_dir
from traceback import format_exc
import sys
from nltk.corpus import stopwords

GO_POS = ["NOUN", "VERB", "ADJ"]
STOP_LIST = [unicode(w) for w in
             ['.', ',', '?', '!', ";", ':', '"', "&", "'", "(", ")", "-", "+", "/", "\\", "|","[","]", "often", "a", "an",
              "-pron-","can", "just"]]
STOP_LIST_NL = [unicode(w) for w in [ 
"de", "van", "een", "en", "het", "in", "is", "op", "te", "met", "voor", "zijn", "dat", "die", "aan", "niet", "om", "ook", "je", "er", "als", "bij", "door", "of", "naar", "uit", "maar", "dan", "over", "ze", "dit", "we", "werd", "al", "wat", "wel", "geen", "zo", "onder", "zal", "gaat", "nieuwe", "waar", "na", "mensen", "twee", "zou", "tussen", "per", "daar", "toch", "heel", "eens", "af", "binnen", "via", "hoe", "zonder", "nodig", "samen", "vanaf", "minder", "gebruikt", "volgens", "bekend", "waarin", "zodat", "allemaal", "vanuit", "wie", "meest", "waardoor"]]

def get_topic_stoplist(preprocess=False):
    stoplist_fpath = join(get_data_dir(), "topic-stoplist-489.csv")
    if preprocess: preprocess_pandas_csv(stoplist_fpath)
    df = read_csv(stoplist_fpath, "\t", encoding='utf8', error_bad_lines=False)
    return [row.word for i,row in df.iterrows()]


def load_stoplist(topic_words=False, lang="en"):
    try:
        if lang == "en":
            if topic_words: return set(get_stop_words("en") + STOP_LIST + get_topic_stoplist())
            else: return set(get_stop_words("en") + STOP_LIST + stopwords.words('english'))
        elif lang == "nl":
            return set(get_stop_words("nl") + stopwords.words('dutch') + STOP_LIST_NL)
    except:
        print "warning: no stopwords were downloaded. check nltk corpora"
        print format_exc()
        return set()


# load resources 
_stop_words = load_stoplist()
print "Loading spacy model..."
_spacy = English()

try:
    _treetagger_fr = TreeTagger(encoding='utf-8',language='french')
    _treetagger_nl = TreeTagger(encoding='utf-8',language='dutch')
    _treetagger_it = TreeTagger(encoding='utf-8',language='italian')
except:
    print "Error: cannot create TreeTagger"
    print format_exc()

def get_stoplist():
    return _stop_words

def lemmatize_tt(text, lang="fr"):
    if lang == "fr": treetagger = _treetagger_fr
    elif lang == "nl": treetagger = _treetagger_nl
    elif lang == "it": treetagger = _treetagger_it

    lemmas = []
    for surface, pos, lemma in treetagger.tag(text):
        if lemma == "<unknown>" or "@" in lemma:
            lemmas.append(surface)
        else:
            lemmas.append(lemma)
    return " ".join(lemmas)

def lemmatize(text, lowercase=True, lang="en"):
    """ Return lemmatized text """
    
    if lang in ["fr", "nl", "it"]:
        text_lemmatized = lemmatize_tt(text, lang=lang)
    else:
        tokens = _spacy(text, tag=True, parse=False, entity=True)
        text_lemmatized = " ".join(t.lemma_ for t in tokens)
    
    if lowercase:
        text_lemmatized = text_lemmatized.lower()

    return text_lemmatized


def add_pos(text):
    """ Add POS tags to input text e.g. 'Car#NOUN is#VERB blue#ADJ.' """
    tokens = _spacy(text, tag=True, parse=False, entity=True)
    return " ".join(t.orth_ + "#" + t.pos_ for t in tokens)


def tokenize(text, pos_filter=False, lowercase=True, remove_stopwords=True, return_pos=False):
    tokens = _spacy(text, tag=True, parse=False, entity=False)
    lemmas = [t.lemma_ for t in tokens if not pos_filter or t.pos_ in GO_POS]
    if remove_stopwords: lemmas = filter(lambda l: l not in _stop_words and l.lower() not in _stop_words, lemmas)
    if lowercase: lemmas = [l.lower() for l in lemmas]

    if return_pos: 
        res = []
        for t in tokens: res.append((t.lemma_, t.pos_))
        return res
    else:
       
        return lemmas

def lemmatize_word(word, lowercase=True):
    try:
        if len(word) == 0: return word
        tokens = _spacy(word, tag=True, parse=False, entity=False)
        if len(tokens) == 0: return word
        lemma = tokens[0].lemma_
        if lowercase: lemma = lemma.lower()
        return lemma
    except KeyboardInterrupt:
         sys.exit()
    except:
        print "Warning: lemmatization error '%s'" % word
        print format_exc()
        return word

def analyze_word(word, lowercase=True):
    tokens = _spacy(word, tag=True, parse=False, entity=False)
    lemma = tokens[0].lemma_
    if lowercase: lemma = lemma.lower()
    return lemma, tokens[0].pos_


def parse(text, pos_filter=False, lowercase=True, remove_stopwords=False):
    tokens = _spacy(text, tag=True, parse=True, entity=False)
    lemmas = [t.lemma_ for t in tokens if not pos_filter or t.pos_ in GO_POS]
    if remove_stopwords: lemmas = filter(lambda l: l not in _stop_words, lemmas)
    if lowercase: lemmas = [l.lower() for l in lemmas]
    return lemmas
