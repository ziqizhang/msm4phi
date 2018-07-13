#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

from sklearn.linear_model import LogisticRegression
from classifier import classifier_learn as cl
from classifier import classifier_tag as ct
from classifier import dataset_loader as dl
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import pandas as pd
from classifier import classifier_util

# Model selection
WITH_SGD = False
WITH_SLR = False
WITH_RANDOM_FOREST = False
WITH_LIBLINEAR_SVM = True
WITH_RBF_SVM = False
WITH_ANN = False

FEATURE_REDUCTION="pca" #str pca or lda or None

# Random Forest model(or any tree-based model) do not ncessarily need feature scaling
N_FOLD_VALIDATION_ONLY = True
SCALING = True
# DIRECTLY LOAD PRE-TRAINED MODEL FOR PREDICTION
# ENABLE THIS VARIABLE TO TEST NEW TEST SET WITHOUT TRAINING
LOAD_MODEL_FROM_FILE = False

# set automatic feature ranking and selection
AUTO_FEATURE_SELECTION = False
FEATURE_SELECTION_WITH_MAX_ENT_CLASSIFIER = False
FEATURE_SELECTION_WITH_EXTRA_TREES_CLASSIFIER = True
FEATURE_SELECTION_MANUAL_SETTING = False
# set manually selected feature index list here
# check random forest setting when changing this variable
MANUAL_SELECTED_FEATURES = []

# The number of CPUs to use to do the computation. -1 means 'all CPUs'
NUM_CPU = -1

N_FOLD_VALIDATION = 5


#####################################################


class Classifer(object):
    """
    supervised org/per pair classifier

    """
    data_feature_file = None
    task_name = None
    identifier = None
    outfolder = None

    def __init__(self, task, identifier, data_X, data_y,
                 outfolder, text_data=None, dnn_embedding_file=None):
        self.test_data = None
        self.training_data = data_X
        self.training_label = data_y
        self.text_data = text_data
        self.identifier = identifier
        self.task_name = task
        self.outfolder = outfolder
        self.dnn_embedding_file = dnn_embedding_file

    def load_testing_data(self, data_test_X):
        self.test_data = data_test_X

    def training(self):
        print("training data size:", len(self.training_data))
        print("train with CPU cores: [%s]" % NUM_CPU)
        # X_resampled, y_resampled = self.under_sampling(self.training_data, self.training_label)
        # Tuning hyper-parameters for precision

        # split the dataset into two parts, 0.75 for train and 0.25 for testing
        X_train = None
        X_test = None
        y_train = None
        y_test = None
        if N_FOLD_VALIDATION_ONLY:
            X_train = self.training_data
            y_train = self.training_label
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_label, test_size=0.25,
                                                                random_state=42)

        ######################### SGDClassifier #######################
        if WITH_SGD:
            cl.learn_generative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "sgd", X_train,
                                y_train,
                                X_test, y_test, self.identifier, self.outfolder,
                                FEATURE_REDUCTION)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            cl.learn_generative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "lr", X_train,
                                y_train,
                                X_test, y_test, self.identifier, self.outfolder,
                                FEATURE_REDUCTION)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "rf", X_train,
                                    y_train,
                                    X_test, y_test, self.identifier, self.outfolder,
                                    FEATURE_REDUCTION)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "svm-l", X_train,
                                    y_train, X_test, y_test, self.identifier, self.outfolder,
                                    FEATURE_REDUCTION)

        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE, "svm-rbf",
                                    X_train,
                                    y_train, X_test, y_test, self.identifier, self.outfolder,
                                    FEATURE_REDUCTION)

        ################# Artificial Neural Network #################
        if WITH_ANN:
            cl.learn_dnn(NUM_CPU, N_FOLD_VALIDATION, self.task_name, LOAD_MODEL_FROM_FILE,
                         self.dnn_embedding_file, self.text_data,
                         X_train,
                         y_train, X_test, y_test, self.identifier, self.outfolder,
                         FEATURE_REDUCTION)

        print("complete!")

    def testing(self):
        print("start testing stage :: testing data size:", len(self.test_data))
        print("test with CPU cores: [%s]" % NUM_CPU)

        ######################### SGDClassifier #######################
        if WITH_SGD:
            ct.tag(NUM_CPU, "sgd", self.task_name, self.test_data)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            ct.tag(NUM_CPU, "lr", self.task_name, self.test_data)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            ct.tag(NUM_CPU, "rf", self.task_name, self.test_data)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            ct.tag(NUM_CPU, "svm-l", self.task_name, self.test_data)
        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            ct.tag(NUM_CPU, "svm-rbf", self.task_name, self.test_data)
        print("complete!")

    def feature_selection_with_max_entropy_classifier(self):
        print("automatic feature selection by maxEnt classifier ...")
        rfe = RFECV(estimator=LogisticRegression(class_weight='auto'),
                    cv=StratifiedKFold(self.training_label, 10), scoring='roc_auc', n_jobs=NUM_CPU)
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

    def feature_selection_with_manual_setting(self):
        print("feature selection with manual setting ...")
        if MANUAL_SELECTED_FEATURES is None or len(MANUAL_SELECTED_FEATURES) == 0:
            raise ArithmeticError("Manual selected feature is NOT set correctly!")

        self.training_data = self.training_data[:, MANUAL_SELECTED_FEATURES]

        print("Optimal number of features : %s" % str(MANUAL_SELECTED_FEATURES))

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
            elif FEATURE_SELECTION_MANUAL_SETTING:
                self.feature_selection_with_manual_setting()
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


# ============= random sampling =================================
# print("training data size before resampling:", len(classifier.training_data))
# X_resampled, y_resampled = classifier.under_sampling(classifier.training_data,                                                         classifier.training_label)
# print("training data size after resampling:", len(X_resampled))
# enable this line to visualise the data
# classifier.training_data = X_resampled
# classifier.training_label = y_resampled

# start the n-fold testing.
        self.training()
# classifier.testing()
