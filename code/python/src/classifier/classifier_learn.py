import functools

import datetime

import gensim
import numpy
from keras import Input, Model
from keras.layers import concatenate, Dense
from keras.preprocessing import sequence
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from classifier import classifier_util as util
import os
from time import time
from classifier import dnn_util as dmc

RANDOM_SEED = 1


def create_feature_reduction_alg(feature_reduction, max_feature=None):
    if feature_reduction == "pca":
        if max_feature is not None and max_feature > 2000:
            return 'pca', PCA(n_components=1000)
        elif max_feature is not None and max_feature < 2000:
            return 'pca', PCA(n_components=int(max_feature / 2))
        else:
            return 'pca', PCA(n_components=1000)
    else:
        return 'lda', LinearDiscriminantAnalysis(n_components=300)


def learn_discriminative(cpus, nfold, task, load_model, model,
                         X_train, y_train, X_test, y_test,
                         identifier, outfolder, feature_reduction=None):
    classifier = None
    model_file = None

    if (model == "rf"):
        print("== Random Forest ...")
        cls = RandomForestClassifier(n_estimators=20, n_jobs=cpus)
        # rfc_tuning_params = {"max_depth": [3, 5, None],
        #                      "max_features": [1, 3, 5, 7, 10],
        #                      "min_samples_split": [2, 5, 10],
        #                      "min_samples_leaf": [1, 3, 10],
        #                      "bootstrap": [True, False],
        #                      "criterion": ["gini", "entropy"]}
        if feature_reduction is not None:
            fr = create_feature_reduction_alg(feature_reduction, len(X_train[0]))
            print("\t using " + str(fr[1]))
            pipe = Pipeline([(fr[0], fr[1]), ('rf', cls)])
            classifier = pipe
        else:
            classifier = cls
        model_file = os.path.join(outfolder, "random-forest_classifier-%s.m" % task)
    if (model == "svm-l"):
        print("== SVM, kernel=linear ...")
        cls = svm.LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge',
                            multi_class='ovr')
        if feature_reduction is not None:
            fr = create_feature_reduction_alg(feature_reduction, len(X_train[0]))
            print("\t using " + str(fr[1]))
            pipe = Pipeline([(fr[0], fr[1]), ('rf', cls)])
            classifier = pipe
        else:
            classifier = cls
        model_file = os.path.join(outfolder, "liblinear-svm-linear-%s.m" % task)

    if (model == "svm-rbf"):
        # tuned_parameters = [{'gamma': np.logspace(-9, 3, 3), 'probability': [True], 'C': np.logspace(-2, 10, 3)},
        #                     {'C': [1e-1, 1e-3, 1e-5, 0.2, 0.5, 1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 2]}]
        print("== SVM, kernel=rbf ...")
        cls = svm.SVC()
        if feature_reduction is not None:
            fr = create_feature_reduction_alg(feature_reduction, len(X_train[0]))
            pipe = Pipeline([(fr[0], fr[1]), ('rf', cls)])
            print("\t using " + str(fr[1]))
            classifier = pipe
        else:
            classifier = cls
        model_file = os.path.join(outfolder, "liblinear-svm-rbf-%s.m" % task)

    nfold_predictions = None

    t0 = time()
    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        classifier = util.load_classifier_model(model_file)
    else:
        # classifier.fit(X_train, y_train)
        nfold_predictions = cross_val_predict(classifier, X_train, y_train, cv=nfold)
        util.save_classifier_model(classifier, model_file)

    if (X_test is not None):
        heldout_predictions_final = classifier.predict(X_test)
        util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test, model, task,
                         identifier, 2, outfolder)
    else:
        util.save_scores(nfold_predictions, y_train, None, y_test, model, task,
                         identifier, 2, outfolder)


def learn_generative(cpus, nfold, task, load_model, model, X_train, y_train, X_test, y_test,
                     identifier, outfolder, feature_reduction=None):
    classifier = None
    model_file = None
    if (model == "sgd"):
        print("== SGD ...")
        # "loss": ["log", "modified_huber", "squared_hinge", 'squared_loss'],
        #               "penalty": ['l2', 'l1'],
        #               "alpha": [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1],
        #               "n_iter": [1000],
        #               "learning_rate": ["optimal"]}
        cls = SGDClassifier(loss='log', penalty='l2', n_jobs=cpus)
        if feature_reduction is not None:
            fr = create_feature_reduction_alg(feature_reduction, len(X_train[0]))
            print("\t using " + str(fr[1]))
            pipe = Pipeline([(fr[0], fr[1]), ('rf', cls)])
            classifier = pipe
        else:
            classifier = cls
        model_file = os.path.join(outfolder, "sgd-classifier-%s.m" % task)
    if (model == "lr"):
        print("== Stochastic Logistic Regression ...")

        cls = LogisticRegression(random_state=111)
        if feature_reduction is not None:
            fr = create_feature_reduction_alg(feature_reduction, len(X_train[0]))
            print("\t using " + str(fr[1]))
            pipe = Pipeline([(fr[0], fr[1]), ('rf', cls)])
            classifier = pipe
        else:
            classifier = cls
        model_file = os.path.join(outfolder, "stochasticLR-%s.m" % task)

    nfold_predictions = None

    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        classifier = util.load_classifier_model(model_file)
    else:
        print(y_train.shape)

        nfold_predictions = cross_val_predict(classifier, X_train, y_train, cv=nfold)

        util.save_classifier_model(classifier, model_file)
    classes = classifier.classes_

    if (X_test is not None):
        heldout_predictions = classifier.predict_proba(X_test)
        heldout_predictions_final = [classes[util.index_max(list(probs))] for probs in heldout_predictions]
        util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test, model, task,
                         identifier, 2, outfolder)
    else:
        util.save_scores(nfold_predictions, y_train, None, y_test, model, task, identifier, 2, outfolder)


def learn_dnn_textonly(cpus, nfold, task, load_model,
                       embedding_model_file,
                       text_data, X_train, y_train, X_test, y_test,
                       identifier, outfolder, feature_reduction=None):
    print("== Perform ANN ...")  # create model

    M = dmc.get_word_vocab(text_data, 1)
    text_based_features = M[0]
    text_based_features = sequence.pad_sequences(text_based_features,
                                                 dmc.DNN_MAX_SEQUENCE_LENGTH)

    gensimFormat = ".gensim" in embedding_model_file
    if gensimFormat:
        pretrained_embedding_models = gensim.models.KeyedVectors.load(embedding_model_file, mmap='r')
    else:
        pretrained_embedding_models = gensim.models.KeyedVectors. \
            load_word2vec_format(embedding_model_file, binary=True)

    pretrained_word_matrix = dmc.build_pretrained_embedding_matrix(M[1],
                                                                   pretrained_embedding_models,
                                                                   dmc.DNN_EMBEDDING_DIM,
                                                                   0)

    create_model_with_args = \
        functools.partial(dmc.create_model, max_index=len(M[1]),
                          wemb_matrix=pretrained_word_matrix, word_embedding_dim=dmc.DNN_EMBEDDING_DIM,
                          max_sequence_length=dmc.DNN_MAX_SEQUENCE_LENGTH,
                          append_feature_matrix=None,
                          model_descriptor=dmc.DNN_MODEL_DESCRIPTOR)

    model = KerasClassifier(build_fn=create_model_with_args, verbose=0, batch_size=100, epochs=10)
    nfold_predictions = cross_val_predict(model,
                                          text_based_features,
                                          y_train,
                                          cv=nfold)

    ann_model_file = os.path.join(outfolder, "ann-%s.m" % task)

    if load_model:
        print("model is loaded from [%s]" % str(ann_model_file))
        best_estimator = util.load_classifier_model(ann_model_file)
    else:
        best_estimator = model

        # self.save_classifier_model(best_estimator, ann_model_file)

    print(datetime.datetime.now())
    print("testing on development set ....")
    if (X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test, model, task, identifier, 2,
                         outfolder)

    else:
        util.save_scores(nfold_predictions, y_train, None, y_test, "dnn", task, identifier, 2, outfolder)

    # util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
    #                       time_ann_predict_dev,
    #                       time_ann_train, y_test)


def learn_dnn_textandmeta(cpus, nfold, task, load_model,
                          embedding_model_file,
                          text_data, X_train_metafeature, y_train, X_test, y_test,
                          identifier, outfolder, prediction_targets):
    print("== Perform ANN ...")  # create model

    M = dmc.get_word_vocab(text_data, 1)
    X_train_textfeature = M[0]
    X_train_textfeature = sequence.pad_sequences(X_train_textfeature,
                                                 dmc.DNN_MAX_SEQUENCE_LENGTH)

    gensimFormat = ".gensim" in embedding_model_file
    if gensimFormat:
        pretrained_embedding_models = gensim.models.KeyedVectors.load(embedding_model_file, mmap='r')
    else:
        pretrained_embedding_models = gensim.models.KeyedVectors. \
            load_word2vec_format(embedding_model_file, binary=True)

    pretrained_word_matrix = dmc.build_pretrained_embedding_matrix(M[1],
                                                                   pretrained_embedding_models,
                                                                   dmc.DNN_EMBEDDING_DIM,
                                                                   2)

    encoder = LabelBinarizer()
    y_train_int = encoder.fit_transform(y_train)

    model_text_inputs = Input(shape=(dmc.DNN_MAX_SEQUENCE_LENGTH,))
    model_text = dmc.create_submodel_textfeature(
                                                 model_text_inputs,
                                                 len(M[1]),
                                                 dmc.DNN_EMBEDDING_DIM, dmc.DNN_MAX_SEQUENCE_LENGTH,
                                                 pretrained_word_matrix,
                                                 dmc.DNN_MODEL_DESCRIPTOR)
    model_metafeature_inputs=Input(shape=(len(X_train_metafeature[0]),))
    model_metafeature = \
        dmc.create_submodel_metafeature(model_metafeature_inputs, 50)

    merge = concatenate([model_text, model_metafeature])
    final = Dense(prediction_targets, activation="softmax")(merge)
    model = Model(inputs=[model_text_inputs, model_metafeature_inputs], outputs=final)

    plot_model(model,to_file="model.png")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    kfold = StratifiedKFold(n_splits=nfold, shuffle=True)
    X_merge=numpy.concatenate([X_train_textfeature, X_train_metafeature],axis=1)

    splits = list(enumerate(kfold.split(X_merge, y_train_int.argmax(1))))

    nfold_predictions = []
    for k in range(0, len(splits)):
        # Fit the model
        X_train_index=splits[k][1][0]
        X_test_index = splits[k][1][1]

        X_train_merge_=X_merge[X_train_index]
        X_test_merge_=X_merge[X_test_index]
        y_train=y_train_int[X_train_index]


        X_train_text_feature= X_train_merge_[:, 0:len(X_train_textfeature[0])]
        X_train_meta_feature=X_train_merge_[:, len(X_train_text_feature[0]):]

        y_test = y_train_int[X_test_index]
        X_test_text_feature = X_test_merge_[:, 0:len(X_train_textfeature[0])]
        X_test_meta_feature = X_test_merge_[:, len(X_train_textfeature[0]):]

        model.fit([X_train_text_feature, X_train_meta_feature],
                  y_train, epochs=dmc.DNN_EPOCHES, batch_size=dmc.DNN_BATCH_SIZE, verbose=2)

        # evaluate the model
        predictions = model.predict([X_test_text_feature,X_test_meta_feature], y_test, verbose=2)
        nfold_predictions.extend(predictions)

        # self.save_classifier_model(best_estimator, ann_model_file)
        util.save_scores(nfold_predictions, y_train, None, None, "dnn", task, identifier, 2, outfolder)

    # util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
    #                       time_ann_predict_dev,
    #                       time_ann_train, y_test)
