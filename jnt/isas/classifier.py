from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import numpy as np
from collections import defaultdict
import traceback
from jnt.common import ensure_dir, exists
from sys import stderr
import os
import json
import codecs
from time import time


D = False  # Debug
CV_FOLDS = 10


class HyperClassifier:
    def __init__(self, model_dir, method="lr", k=100):

        self.CLASSIFIER_FILE = "classifier.pkl"
        self.VOC_FILE = "voc.csv"
        self.KBEST_VOC_FILE = "kbest-voc.csv"
        self.VEC_FILE = "vectorizer.pkl"
        self.KBEST_FILE = "kbest.pkl"
        self.META_FILE = "meta.json"
        clf_fpath = os.path.join(model_dir, self.CLASSIFIER_FILE)
        voc_fpath = os.path.join(model_dir, self.VEC_FILE)
        kbest_fpath = os.path.join(model_dir, self.KBEST_FILE)
        self._model_dir = model_dir
        self._meta_fpath = os.path.join(model_dir, self.META_FILE)

        self._meta = {}
        self._meta["method"] = method
        self._meta["k"] = k

        if exists(model_dir) and exists(clf_fpath) and exists(voc_fpath) and exists(kbest_fpath):
            # load the model
            self._clf = joblib.load(clf_fpath)
            self._vec = joblib.load(voc_fpath)
            self._kbest = joblib.load(kbest_fpath)
            self._meta = json.load(open(self._meta_fpath, "r"))
            print >> stderr, "Metadata were loaded from:", self._meta_fpath
        else:
            # model doesn't exist, create a new one
            ensure_dir(model_dir)
            self.save_meta()

    def __str__(self):
        return "Classifier:" + str(self._meta)

    def init_preprocessors(self):
        self._clf = None
        self._kbest = SelectKBest(chi2, k=self._meta["k"])

    def save_meta(self):
        json.dump(self._meta, open(self._meta_fpath, "w"))
        print >> stderr, "Meta file saved to:", self._meta_fpath

    def save_voc(self, X, output_fpath):
        with codecs.open(output_fpath, "w", "utf-8") as voc_file:
            if D: print >> stderr, "\nfeatures weights:"
            f = self._vec.get_feature_names()
            w = np.asarray(X.sum(axis=0)).ravel()
            pairs = zip(f, w)
            pairs.sort(key=lambda tup: tup[1],reverse=True)
            for x in pairs:
                #print "%s;%s\n" % (x[0], x[1])
                voc_file.write("%s;%s\n" % (x[0], x[1]))
                if D: print >> stderr, x[0], x[1]

            print >> stderr, "Saved vocabulary to:", output_fpath


    def save_features(self, output_fpath):
        tic = time()
        print >> stderr, "Saving features to file %s..." % output_fpath

        bow_names = np.asarray(self._vec.get_feature_names())[self._kbest.get_support()]
        bow_names = [fn for fn in bow_names]
        self._meta["bow_features"]: feature_names = bow_names
        print >> stderr, "feature names len:", len(feature_names)

        features = defaultdict(list)
        for k, c in enumerate(self._clf.classes_):
            if k > len(self._clf.coef_)-1:
                continue  # when we have two classes (binary classification) there will be just one feature vector
            for i, fn in enumerate(feature_names):
                features[c].append( (self._clf.coef_[k][i], fn) )

        with codecs.open(output_fpath, "w", "utf-8") as output_file:
            for c in features:
                for f in sorted(features[c], reverse=True):
                    try:
                        print >> output_file, "%s;%s;%f" % (c, f[1], f[0])
                    except:
                        print >> stderr, "Bad feature:", c, f
                        print >> stderr, traceback.format_exc()

        print >> stderr, "Features are saved to file %s in %d sec." % (output_fpath, time() - tic)

    def getBOWFeatures(self, texts, train_mode):
        X = None
        if self._meta["bow_features"]:
            print >> stderr, "getBOW..."
            tac = time()
            if train_mode:
                # Get and save the initial vocabulary
                X = self._vec.fit_transform(texts)
                joblib.dump(self._vec, os.path.join(self._model_dir, self.VEC_FILE))
                print >> stderr, "Saved vectorizer to:", os.path.join(self._model_dir, self.VEC_FILE)
            else:
                X = self._vec.transform(texts)
            tuk = time()
            print >> stderr, "getBOW: %d sec." % (tuk - tac)
        return X


    def selectBOWFeatures(self, X, train_mode, y):
        X_bow = None
        if self._meta["bow_features"]:
            print >> stderr, "selectBOW..."
            tac = time()
            if train_mode:
                # Get and save a reduced vocabulary
                X_bow = self._kbest.fit_transform(X, y)
                joblib.dump(self._kbest, os.path.join(self._model_dir, self.KBEST_FILE))
                print >> stderr, "Saved k-best to:", os.path.join(self._model_dir, self.KBEST_FILE)
            else:
                X_bow = self._kbest.transform(X)
            tuk = time()
            print >> stderr, "selectBOW: %d sec." % (tuk - tac)
            print >> stderr, "X_bow.shape:", X_bow.shape
        return X_bow


    def get_features(self, texts, y=None, train_mode=False):
        print >> stderr, "Feature extraction..."
        tic=time()

        X = self.getBOWFeatures(texts, train_mode)
        if train_mode and self._meta["bow_features"]:
            self.save_voc(X, os.path.join(self._model_dir, self.VOC_FILE))

        X_bow = self.selectBOWFeatures(X, train_mode, y)
        X_dict = self.getDictFeatures(texts)

        X_fin = self.getFinalFeatures(X_bow, X_dict)

        toc=time()
        print >> stderr, "Feature extraction: %d sec." % (toc-tic)
        return X_fin


    def train(self, dataset_fpath):
        self.init_preprocessors()
        fdict, texts, labels = self.load_dataset(dataset_fpath)
        self._clf = self.train_texts(texts, labels)

        self._meta["dataset"] = dataset_fpath
        self.save_meta()

        return self._clf


    def train_texts(self, texts, y):
        # Train
        X = self.get_features(texts, y, train_mode=True)
        clf = self.train_features(X, y)

        # Print stat
        if D:
            print >> stderr, "\ncoef:", clf.coef_
            print >> stderr, "accuracy:", clf.score(X, y)
            print >> stderr, "class log prior:"
            for i, v in enumerate(clf.class_log_prior_):
                print >> stderr, i, math.exp(v)

        # Save the model
        clf_fpath = os.path.join(self._model_dir, self.CLASSIFIER_FILE)
        joblib.dump(clf, clf_fpath)
        print >> stderr, "Saved model to:", clf_fpath
        self._clf = clf
        self.save_features(os.path.join(self._model_dir, self.KBEST_VOC_FILE))

        return clf


    def train_features(self, X, y):
        print >> stderr, "Training classifier..."
        tic=time()

        if self._meta["method"] == "lr": clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, class_weight='auto')
        elif self._meta["method"] == "nb": clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        elif self._meta["method"] == 'linearsvc': clf = LinearSVC(C=1.0)
        elif self._meta["method"] == 'svc':  clf = SVC(C=1.0, probability=True)
        elif self._meta["method"] == 'svc-linear': clf = SVC(kernel='linear', C=1.0, probability=True)
        elif self._meta["method"] == 'rfc': clf = RandomForestClassifier(n_estimators = 100)
        elif self._meta["method"] == 'dummy': clf = DummyClassifier(strategy='most_frequent', random_state=0)
        else: clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, class_weight='auto')
        c = clf.fit(X, y)

        toc=time()
        print >> stderr, "Traning classifier: %d sec." % (toc-tic)
        return c

    def predict_proba(self, clf, X):
        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X)  # n_samples x n_classes probabilities matrix
        elif hasattr(clf, '_predict_proba_lr'):
            print >> stderr, "WARNING: Classifier %r doesn't have method 'predict_proba', using '_predict_proba_lr' instead." % clf
            scores = clf._predict_proba_lr(X)
        else:
            raise TypeError("Classifier %r doesn't have doesn't have 'predict_proba' or '_predict_proba_lr' methods!" % clf)
        return scores

    def evaluate(self, dataset_fpath, test_size=0.3, aggressive_cv=False):
        # load data
        X, y = self.vectorize(dataset_fpath)
        #X = self._vec.fit_transform(texts)
        #X = self._kbest.fit_transform(X, y)

        # build classification report
        X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, test_size=test_size, random_state=0)
        clf = self.train_features(X_tr, y_tr)

        try:
            self._clf = clf
            self.save_features(os.path.join(self._model_dir, self.KBEST_VOC_FILE))
        except Exception:
            print >> stderr, "WARNING: Error occurred when saving features to file; the features were not saved as plaintext so you cannot look at them!"
            traceback.print_exc()

        y_tt_pred = clf.predict(X_tt)
        print >> stderr, classification_report(y_tt, y_tt_pred)

        try:
            classification_utils.save_confusion_matrix(y_tt, y_tt_pred, self._model_dir)
        except Exception:
            print >> stderr, "WARNING: Error occurred when saving confusion matrix: matrix wasn't saved!"
            traceback.print_exc()

        try:
            self.save_accuracy_coverage_matrix(y_tt, X_tt, clf)
        except Exception:
            print >> stderr, "WARNING: Error occurred when saving accuracy-coverage matrix: matrix wasn't saved!"
            traceback.print_exc()

        # try:
        #     self.save_precision_recall_curve(y_tt, y_tt_pred)
        # except Exception:
        #     print >> stderr, "WARNING: Error occurred when saving precision-recall curve: curve wasn't saved!"
        #     traceback.print_exc()

        # cv
        m = "accuracy"
        y = np.array(y)
        cv_jobs = -1 if aggressive_cv else 1
        s = cross_val_score(clf, X, y, cv=CV_FOLDS, scoring=m, n_jobs=cv_jobs)
        print >> stderr, "\t%s: %.2f +- %.2f" % (m, s.mean(), s.std())

        try:
            print >> stderr, "Plotting learning curve..."
            tac = time()
            classification_utils.plot_learning_curve(clf, repr(clf), X, y, ylim=(0.3, 1.01), cv=CV_FOLDS,
                                                     outdir=self._model_dir)
            print >> stderr, "Learning curve plotted: %d sec." % (time()-tac)
        except Exception:
            print >> stderr, "WARNING: Error occurred when plotting learning curves: learning curves were not plotted!"
            traceback.print_exc()


    def accuracy(self, X, y):
        a = self._clf.score(X, y)
        return a


    def accuracy_dataset(self, dataset_fpath):
        X, y = self.get_Xy(dataset_fpath)
        a = self.accuracy(X, y)
        print >> stderr, "Score on:", dataset_fpath
        print >> stderr, "\taccuracy:", a
        return a


    def format_probs(self, probs):
        # format results
        labels = []
        confs = []
        for p in probs:
            j_best = np.argmax(p)
            labels.append(self._clf.classes_[j_best])
            confs.append(p[j_best])

        return labels, confs, probs, self._clf.classes_


    def classify(self, dataset_fpath, output_fpath=None, n_best=3):
        # perform classification
        fdict, texts, labels = self.load_dataset(dataset_fpath)
        labels_pred, confs, probs, classes = self.classify_texts(texts)
        nlabels = self.calculate_nlabels(probs, classes, n_best, output_fpath)

        return nlabels, labels, confs, probs, classes


    def calculate_nlabels(self, probs, classes, n_best, output_fpath=None):
        # calculate n best labels
        nlabels = []

        for i, x in enumerate(probs):
            nbest = []
            try:
                maxs = probs[i].argsort()[-n_best:][::-1]
                for j in maxs:
                    nbest.append((classes[j], probs[i][j]))
            finally:
                nlabels.append(nbest)

        if len(nlabels) != len(probs):
            print >> stderr, "Warning: dimensions of nlabels and probs are not equal."

        # save results to file if needed
        if output_fpath is not None:
            with codecs.open(output_fpath, "w", "utf-8") as out_file:
                print >> out_file, "conf;label_pred;label_orig;text"
                if len(labels) == len(labels_pred):
                    for i, t in enumerate(texts):
                        print >> out_file, "%.3f;%s;%s;%s" % (
                                confs[i], labels_pred[i], labels[i], texts[i].decode("utf-8"))
                else:
                    print >> stderr, "Warning: len(labels) != len(labels_pred):", len(labels), len(labels_pred)

        return nlabels


    def classify_texts(self, texts):
        if self._clf == None:
            print >> stderr, "Error: no model. Train or load a model."
            return
        X = self.get_features(texts, y=None, train_mode=False)
        probs = self.predict_proba(self._clf, X)
        return self.format_probs(probs)


