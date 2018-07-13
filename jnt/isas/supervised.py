from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomTreesEmbedding, BaggingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import numpy as np
from collections import defaultdict
import traceback
from jnt.common import ensure_dir, exists
from os.path import join
import json
import codecs
from time import time
from traceback import format_exc
from pandas import Series
from sklearn.grid_search import GridSearchCV


CV_NUM = 5
FEATURES = ["hyper_in_hypo_i","hypo2hyper_substract"]
METHODS = ["MLP"]
 #"LogisticRegressionL2","SVCLinear","LinearSVC""GradientBoosting","Bagging","RandomForest", "MultinomialNB", "Dummy","AdaBoost","LogisticRegressionL1"]


class SuperTaxi:
    def __init__(self, model_dir, method="LogisticRegressionL2", features=FEATURES, k=100, overwrite=False):

        self.CLASSIFIER_FILE = "classifier"
        self.KBEST_VOC_FILE = "kbest-voc.csv"
        self.KBEST_FILE = "kbest.pkl"
        self.META_FILE = "meta.json"
        clf_fpath = join(model_dir, self.CLASSIFIER_FILE)
        kbest_fpath = join(model_dir, self.KBEST_FILE)
        self._model_dir = model_dir
        self._meta_fpath = join(model_dir, self.META_FILE)

        self._meta = {}
        self._meta["method"] = method
        self._meta["k"] = k
        self._meta["features"] = features

        if exists(model_dir) and exists(clf_fpath) and not overwrite:
            # load the model
            self._clf = joblib.load(clf_fpath)
            self._meta = json.load(open(self._meta_fpath, "r"))
            print("Metadata were loaded from:", self._meta_fpath)
        else:
            # model doesn't exist, or must be overwritten create a new one
            ensure_dir(model_dir)
            self.save_meta()

    @property
    def meta(self):
        return self._meta

    def __str__(self):

        try: params = self._clf.get_params()
        except: params = ""
        return "Classifier:" + str(self._meta) + "\n" + str(params)

    def init_preprocessors(self):
        self._clf = None
        self._kbest = SelectKBest(chi2, k=self._meta["k"])

    def save_meta(self):
        json.dump(self._meta, open(self._meta_fpath, "w"))
        print("Meta file saved to:", self._meta_fpath)

    def save_voc(self, X, output_fpath):
        with codecs.open(output_fpath, "w", "utf-8") as voc_file:
            if D: print("\nfeatures weights:")
            f = self._vec.get_feature_names()
            w = np.asarray(X.sum(axis=0)).ravel()
            pairs = list(zip(f, w))
            pairs.sort(key=lambda tup: tup[1],reverse=True)
            for x in pairs:
                #print "%s;%s\n" % (x[0], x[1])
                voc_file.write("%s;%s\n" % (x[0], x[1]))
                if D: print(x[0], x[1])

            print("Saved vocabulary to:", output_fpath)

    def save_features(self, output_fpath):
        tic = time()
        print("Saving features to file %s..." % output_fpath)

        bow_names = np.asarray(self._vec.get_feature_names())[self._kbest.get_support()]
        bow_names = [fn for fn in bow_names]
        feature_names = bow_names
        print("feature names len:", len(feature_names))

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
                        print("%s;%s;%f" % (c, f[1], f[0]), file=output_file)
                    except:
                        print("Bad feature:", c, f)
                        print(traceback.format_exc())

        print("Features are saved to file %s in %d sec." % (output_fpath, time() - tic))

    @property
    def classifier_fpath(self):
        return join(self._model_dir, self.CLASSIFIER_FILE)

    def _relations2features(self, relations):
        df = relations.copy()
        X_names = self._meta["features"]
        y_name = "correct"
        df_X = df[X_names]
        X = df_X.as_matrix()

        df_y = df[y_name]
        y = df_y.as_matrix()
        print(X.shape, y.shape)

        return X, y, X_names, y_name

    def crossval(self, relations):
        X, y, X_names, y_name = self._relations2features(relations)
        clf = self._create_classifier()
        for scoring in ["precision", "recall", "f1", "accuracy"]:
            s = cross_val_score(clf, X, y, cv=CV_NUM, scoring=scoring, n_jobs=CV_NUM)
            print("\t%s: %.2f +- %.2f" % (scoring, s.mean(), s.std()))

    def train(self, relations):
        self.X, self.y, self.X_names, self.y_name = self._relations2features(relations)
        self._clf = self._train_classifier(self.X, self.y)
        joblib.dump(self._clf, self.classifier_fpath)
        print("Saved classifier to:", self.classifier_fpath)

        return self._clf

    def _train_classifier(self, X, y):
        print("Training classifier...")
        tic=time()

        clf = self._create_classifier()
        c = clf.fit(X, y)

        toc=time()
        print("Traning classifier: %d sec." % (toc-tic))
        return c

    def grid_search_svc(self, relations, test=False):
        if test:
            param_grid = [
              {'C': [0.1, 10, 100], 'kernel': ['linear']},
              {'C': [10], 'gamma': [1, 0.01], 'kernel': ['rbf']},
            ]
        else:
            param_grid = [
              {'C': [0.1, 1, 10, 50, 100, 1000, 10000], 'kernel': ['linear']},
              {'C': [0.1, 1, 10, 50, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
            ]

        e = SVC(C=1, kernel='linear', probability=False)
        clf = GridSearchCV(estimator=e, param_grid=param_grid, scoring=None,
                     fit_params=None, n_jobs=16, iid=True, refit=True, cv=CV_NUM,
                     verbose=0, pre_dispatch='2*n_jobs', error_score='raise')

        X, y, X_names, y_name = self._relations2features(relations)

        #if test:
        #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        #    X = X_test
        #    y = y_test

        self._clf = clf.fit(X, y)

        print("Best score:", self._clf.best_score_)
        print("Best classifier:", self._clf.best_estimator_)
        joblib.dump(self._clf, self.classifier_fpath)
        print("Saved best classifier to:", self.classifier_fpath)

        return

    def predict(self, relations):
        if self._clf == None:
            print("Error: classifier is not loaded.")
            return relations

        X, y, X_names, y_name = self._relations2features(relations)

        if hasattr(self._clf, 'predict_proba'):
            conf = self._clf.predict_proba(X)
        else:
            conf = np.ones([len(relations), 2])

        y_predict = self._clf.predict(X)
        relations["correct_predict"] = Series(y_predict, index=relations.index, dtype=float)
        relations["correct_predict_conf"] = Series(np.max(conf, axis=1), index=relations.index, dtype=float)

        return relations

    def _create_classifier(self):
        if self._meta["method"] == "LogisticRegressionL2": clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)  #, class_weight='auto')
        elif self._meta["method"] == "LogisticRegressionL1": clf = LogisticRegression(penalty='l1', tol=0.0001, C=1.0)  #, class_weight='auto')
        elif self._meta["method"] == "MultinomialNB": clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        elif self._meta["method"] == 'LinearSVC': clf = LinearSVC(C=1.0)
        elif self._meta["method"] == 'SVC':  clf = SVC(C=1.0, probability=True)
        elif self._meta["method"] == 'SVCLinear': clf = SVC(kernel='linear', C=1.0, probability=True)
        elif self._meta["method"] == 'RandomForest': clf = RandomForestClassifier(n_estimators = 100)
        elif self._meta["method"] == 'Dummy': clf = DummyClassifier(strategy='most_frequent', random_state=0)
        elif self._meta["method"] == 'GradientBoosting': clf = GradientBoostingClassifier()
        elif self._meta["method"] == 'AdaBoost': clf = AdaBoostClassifier()
        elif self._meta["method"] == 'RandomTreesEmbedding': clf = RandomTreesEmbedding()
        elif self._meta["method"] == 'Bagging': clf = BaggingClassifier()
        elif self._meta["method"] == 'MLP': clf = MLPClassifier(alpha=1)
        else: clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)  # , class_weight='auto')
        return clf

    def _print_clf_info(self):
        try:
            print("parameters:", self._clf.get_params())
            y_predict = self._clf.predict(self.X)
            print("True:", self.y)
            print("Predicted:\n", y_predict)
            print("Data:\n", self.X)
            print(classification_report(self.y, y_predict))
            print("score:", self._clf.score(self.X, self.y))

            if hasattr(self._clf, 'coef_'): print("weights:", self._clf.coef_)
            if hasattr(self._clf, 'classes_'):  print("classes:", self._clf.classes_)
            if hasattr(self._clf, 'class_weights_'): print("class weights:", self._clf.class_weight)
        except:
            print(format_exc())
