#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['THEANO_FLAGS'] = "floatX=float64,device=cpu,openmp=True"
# # os.environ['THEANO_FLAGS']="openmp=True"
# os.environ['OMP_NUM_THREADS'] = '16'

from sklearn.linear_model import LogisticRegression
from classifier import classifier_learn as cl
from classifier import classifier_tag as ct
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from classifier import classifier_util

# Random Forest model(or any tree-based model) do not ncessarily need feature scaling
SCALING = True
# DIRECTLY LOAD PRE-TRAINED MODEL FOR PREDICTION
# ENABLE THIS VARIABLE TO TEST NEW TEST SET WITHOUT TRAINING
LOAD_MODEL_FROM_FILE = False

# set automatic feature ranking and selection
AUTO_FEATURE_SELECTION = False
FEATURE_SELECTION_WITH_MAX_ENT_CLASSIFIER = False
FEATURE_SELECTION_WITH_EXTRA_TREES_CLASSIFIER = True
FEATURE_SELECTION_MANUAL_SETTING = False

#####################################################


class Classifer(object):
    """
    supervised org/per pair classifier

    """
    data_feature_file = None
    task_name = None
    identifier = None
    outfolder = None

    '''
    text_data=text content per instance. must have the same number of rows as data_X.
                if provided, features will be extracted for these text, and 
                concatenated with data_X
     
    dnn_embedding_file=if dnn model is used, this must be a file pointing to the embedding model
    dnn_descriptor=if dnn model is used, this must be a text description of the model
                    see dnn_util.py. There are only a limited set of descriptors that can be
                    parsed
    *algorithms=sgd,svm_l,svm_rbf,rf,lr,dnn_text (which uses only text data), dnn (which concatenates
                features from text with meta features). For both dnn and dnn_text, text_data
                must not be None. For others, if 'pca-' is added as prefix (e.g., pca-sgd), 
                PCA will be applied for the feature space before the algorithm runs
    '''

    def __init__(self, task, identifier, data_X, data_y,
                 outfolder, categorical_targets, algorithms:list,
                 nfold=10, text_data=None,
                 dnn_embedding_file=None,
                 dnn_descriptor=None, ):
        self.test_data = None
        self.training_data = data_X
        self.training_label = data_y
        self.nfold = nfold
        self.categorical_targets = categorical_targets
        self.text_data = text_data
        self.identifier = identifier
        self.task_name = task
        self.outfolder = outfolder
        self.dnn_embedding_file = dnn_embedding_file
        self.dnn_descriptor = dnn_descriptor
        self.algorithms = algorithms

    def load_testing_data(self, data_test_X):
        self.test_data = data_test_X

    def training(self):
        print("training data size:", len(self.training_data))
        # X_resampled, y_resampled = self.under_sampling(self.training_data, self.training_label)
        # Tuning hyper-parameters for precision

        # split the dataset into two parts, 0.75 for train and 0.25 for testing
        X_train = self.training_data
        y_train = self.training_label
        X_test = None
        y_test = None

        ######################### SGDClassifier #######################
        if "sgd" in self.algorithms:
            cl.learn_generative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "sgd", X_train,
                                y_train,
                                X_test, y_test, self.identifier, self.outfolder)
        if "pca-sgd" in self.algorithms:
            cl.learn_generative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "sgd", X_train,
                                y_train,
                                X_test, y_test, self.identifier, self.outfolder, "pca")

        ######################### Stochastic Logistic Regression#######################
        if "lr" in self.algorithms:
            cl.learn_generative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "lr", X_train,
                                y_train,
                                X_test, y_test, self.identifier, self.outfolder)
        if "pca-lr" in self.algorithms:
            cl.learn_generative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "lr", X_train,
                                y_train,
                                X_test, y_test, self.identifier, self.outfolder, "pca")

        ######################### Random Forest Classifier #######################
        if "rf" in self.algorithms:
            cl.learn_discriminative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "rf", X_train,
                                    y_train,
                                    X_test, y_test, self.identifier, self.outfolder)
        if "pca-rf" in self.algorithms:
            cl.learn_discriminative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "rf", X_train,
                                    y_train,
                                    X_test, y_test, self.identifier, self.outfolder,"pca")

        ###################  liblinear SVM ##############################
        if "svm_l" in self.algorithms:
            cl.learn_discriminative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "svm-l", X_train,
                                    y_train, X_test, y_test, self.identifier, self.outfolder)
        if "pca-svm_l" in self.algorithms:
            cl.learn_discriminative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "svm-l", X_train,
                                    y_train, X_test, y_test, self.identifier, self.outfolder, "pca")

        ##################### RBF svm #####################
        if "svm_rbf" in self.algorithms:
            cl.learn_discriminative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "svm-rbf",
                                    X_train,
                                    y_train, X_test, y_test, self.identifier, self.outfolder)
        if "pca-svm_rbf" in self.algorithms:
            cl.learn_discriminative(-1, self.nfold, self.task_name, LOAD_MODEL_FROM_FILE, "svm-rbf",
                                    X_train,
                                    y_train, X_test, y_test, self.identifier, self.outfolder, "pca")

        ################# Artificial Neural Network #################
        if "dnn_text" in self.algorithms:
            cl.learn_dnn_textonly(self.nfold, self.task_name, LOAD_MODEL_FROM_FILE,
                                  self.dnn_embedding_file, self.text_data,
                                  X_train,
                                  y_train, X_test, y_test, self.dnn_descriptor, self.outfolder)

        if "dnn" in self.algorithms:
            cl.learn_dnn_textandmeta(self.nfold, self.task_name, LOAD_MODEL_FROM_FILE,
                                     self.dnn_embedding_file, self.text_data,
                                     X_train,
                                     y_train, X_test, y_test, self.dnn_descriptor, self.outfolder,
                                     prediction_targets=self.categorical_targets)

        print("complete!")

    def testing(self):
        print("start testing stage :: testing data size:", len(self.test_data))

        ######################### SGDClassifier #######################
        if "sgd" in self.algorithms:
            ct.tag(-1, "sgd", self.task_name, self.test_data)

        ######################### Stochastic Logistic Regression#######################
        if "lr" in self.algorithms:
            ct.tag(-1, "lr", self.task_name, self.test_data)

        ######################### Random Forest Classifier #######################
        if "rf" in self.algorithms:
            ct.tag(-1, "rf", self.task_name, self.test_data)

        ###################  liblinear SVM ##############################
        if "svm-l" in self.algorithms:
            ct.tag(-1, "svm-l", self.task_name, self.test_data)
        ##################### RBF svm #####################
        if "svm-rbf" in self.algorithms:
            ct.tag(-1, "svm-rbf", self.task_name, self.test_data)
        print("complete!")

    def feature_selection_with_max_entropy_classifier(self):
        print("automatic feature selection by maxEnt classifier ...")
        rfe = RFECV(estimator=LogisticRegression(class_weight='auto'),
                    cv=StratifiedKFold(self.training_label, 10), scoring='roc_auc', n_jobs=-1)
        rfe.fit(self.training_data, self.training_label)

        self.training_data = rfe.transform(self.training_data)
        print("Optimal number of features : %d" % rfe.n_features_)

    def feature_selection_with_extra_tree_classifier(self):
        print("feature selection with extra tree classifier ...")
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        clf = ExtraTreesClassifier()
        clf = clf.fit(self.training_data, self.training_label)

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1].tolist()
        model = SelectFromModel(clf, prefit=True)
        X_n = model.transform(self.training_data).shape[1]
        features_selected = indices[0:X_n]
        features_selected.sort()

        self.training_data = self.training_data[:, features_selected]

        print("Optimal number of features : %s" % str(features_selected))

    def saveOutput(self, prediction, model_name):
        filename = os.path.join(os.path.dirname(__file__), "prediction-%s-%s.csv" % (model_name, self.task_name))
        file = open(filename, "w")
        for entry in prediction:
            if (isinstance(entry, float)):
                file.write(str(entry) + "\n")
                # file.write("\n")
            else:
                if (entry[0] > entry[1]):
                    file.write("0\n")
                else:
                    file.write("1\n")
        file.close()

    def run(self):
        # classifier.load_testing_data(DATA_ORG)
        classifier_util.validate_training_set(self.training_data)

        if AUTO_FEATURE_SELECTION:  # this is false by default
            if FEATURE_SELECTION_WITH_EXTRA_TREES_CLASSIFIER:
                self.feature_selection_with_extra_tree_classifier()
            elif FEATURE_SELECTION_WITH_MAX_ENT_CLASSIFIER:
                self.feature_selection_with_max_entropy_classifier()
            else:
                raise ArithmeticError("Feature selection method IS NOT SET CORRECTLY!")

        # ============== feature scaling =====================
        if SCALING:
            print("feature scaling method: standard dev")

            self.training_data = classifier_util.feature_scaling_mean_std(self.training_data)
            if self.test_data is not None:
                self.test_data = classifier_util.feature_scaling_mean_std(self.test_data)

            # print("example training data after scaling:", classifier.training_data[0])

        else:
            print("training without feature scaling!")

        self.training()
