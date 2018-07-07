import functools

import datetime
import gensim
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

from classifier import classifier_util as util
from sklearn.model_selection import GridSearchCV
import os
from time import time
from classifier import dnn_util as du


def learn_discriminative(cpus, nfold, task, load_model, model, X_train, y_train, X_test, y_test,
                         identifier, outfolder):
    classifier = None
    model_file = None

    if (model == "rf"):
        print("== Random Forest ...")
        classifier = RandomForestClassifier(n_estimators=20, n_jobs=cpus)
        # rfc_tuning_params = {"max_depth": [3, 5, None],
        #                      "max_features": [1, 3, 5, 7, 10],
        #                      "min_samples_split": [2, 5, 10],
        #                      "min_samples_leaf": [1, 3, 10],
        #                      "bootstrap": [True, False],
        #                      "criterion": ["gini", "entropy"]}
        rfc_tuning_params = {}
        classifier = GridSearchCV(classifier, param_grid=rfc_tuning_params, cv=nfold,
                                  n_jobs=cpus)
        model_file = os.path.join(outfolder, "random-forest_classifier-%s.m" % task)
    if (model == "svm-l"):
        tuned_parameters = [{'C': [0.01]},
                            {'C': [0.01]}]
        # tuned_parameters = [{'C': [0.01]}]

        print("== SVM, kernel=linear ...")
        classifier = svm.LinearSVC(class_weight='balanced', C=0.01, penalty='l2', loss='squared_hinge',
                                   multi_class='ovr')
        classifier = GridSearchCV(classifier, tuned_parameters[0], cv=nfold, n_jobs=cpus)
        model_file = os.path.join(outfolder, "liblinear-svm-linear-%s.m" % task)

    if (model == "svm-rbf"):
        # tuned_parameters = [{'gamma': np.logspace(-9, 3, 3), 'probability': [True], 'C': np.logspace(-2, 10, 3)},
        #                     {'C': [1e-1, 1e-3, 1e-5, 0.2, 0.5, 1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 2]}]
        tuned_parameters = [{'C': [0.01]},
                            {'C': [0.01]}]
        print("== SVM, kernel=rbf ...")
        classifier = svm.SVC()
        classifier = GridSearchCV(classifier, param_grid=tuned_parameters[0], cv=nfold, n_jobs=cpus)
        model_file = os.path.join(outfolder, "liblinear-svm-rbf-%s.m" % task)

    best_param = []
    cv_score = 0
    best_estimator = None
    nfold_predictions = None

    t0 = time()
    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        best_estimator = util.load_classifier_model(model_file)
    else:
        classifier.fit(X_train, y_train)
        nfold_predictions = cross_val_predict(classifier.best_estimator_, X_train, y_train, cv=nfold)

        best_estimator = classifier.best_estimator_
        best_param = classifier.best_params_
        cv_score = classifier.best_score_
        util.save_classifier_model(best_estimator, model_file)

    if (X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test, model, task,
                         identifier, 2, outfolder)
    else:
        util.save_scores(nfold_predictions, y_train, None, y_test, model, task,
                         identifier, 2, outfolder)


def learn_generative(cpus, nfold, task, load_model, model, X_train, y_train, X_test, y_test,
                     identifier, outfolder):
    classifier = None
    model_file = None
    if (model == "sgd"):
        print("== SGD ...")
        sgd_params = {}
        # "loss": ["log", "modified_huber", "squared_hinge", 'squared_loss'],
        #               "penalty": ['l2', 'l1'],
        #               "alpha": [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1],
        #               "n_iter": [1000],
        #               "learning_rate": ["optimal"]}
        classifier = SGDClassifier(loss='log', penalty='l2', n_jobs=cpus)

        classifier = GridSearchCV(classifier, param_grid=sgd_params, cv=nfold,
                                  n_jobs=cpus)
        model_file = os.path.join(outfolder, "sgd-classifier-%s.m" % task)
    if (model == "lr"):
        print("== Stochastic Logistic Regression ...")
        slr_params = {"penalty": ['l2'],
                      "solver": ['liblinear'],
                      # "C": list(np.power(10.0, np.arange(-10, 10))),
                      "max_iter": [10000]}
        classifier = LogisticRegression(random_state=111)
        classifier = GridSearchCV(classifier, param_grid=slr_params, cv=nfold,
                                  n_jobs=cpus)
        model_file = os.path.join(outfolder, "stochasticLR-%s.m" % task)

    best_param = []
    cv_score = 0
    best_estimator = None
    nfold_predictions = None

    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        best_estimator = util.load_classifier_model(model_file)
    else:
        print(y_train.shape)

        classifier.fit(X_train, y_train)
        nfold_predictions = cross_val_predict(classifier.best_estimator_, X_train, y_train, cv=nfold)

        best_estimator = classifier.best_estimator_
        best_param = classifier.best_params_
        cv_score = classifier.best_score_
        util.save_classifier_model(best_estimator, model_file)
    classes = classifier.best_estimator_.classes_

    if (X_test is not None):
        heldout_predictions = best_estimator.predict_proba(X_test)
        heldout_predictions_final = [classes[util.index_max(list(probs))] for probs in heldout_predictions]
        util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test, model, task,
                         identifier, 2, outfolder)
    else:
        util.save_scores(nfold_predictions, y_train, None, y_test, model, task, identifier, 2, outfolder)


def learn_dnn(cpus, nfold, task, load_model,
              embedding_model_file,
              text_data, X_train, y_train, X_test, y_test,
              identifier, outfolder):
    print("== Perform ANN ...")  # create model

    M = du.get_word_vocab(text_data, 1)
    text_based_features=M[0]
    text_based_features = sequence.pad_sequences(text_based_features, maxlen=100)

    gensimFormat = ".gensim" in embedding_model_file
    if gensimFormat:
        pretrained_embedding_models=gensim.models.KeyedVectors.load(embedding_model_file, mmap='r')
    else:
        pretrained_embedding_models=gensim.models.KeyedVectors. \
                          load_word2vec_format(embedding_model_file, binary=True)

    pretrained_word_matrix = du.build_pretrained_embedding_matrix(M[1],
                                                               pretrained_embedding_models,
                                                               300,
                                                               0)

    create_model_with_args = \
        functools.partial(create_model, max_index=len(M[1]),
                          wemb_matrix=pretrained_word_matrix,
                          append_feature_matrix=None,
                          model_descriptor="f_sub_conv[1,2,3](conv1d=100),maxpooling1d=4,flatten,dense=6-softmax")
    model = KerasClassifier(build_fn=create_model_with_args, verbose=0)

    # model = KerasClassifier(build_fn=create_model_with_args, verbose=0, batch_size=100,
    #                         nb_epoch=10)
    #
    # nfold_predictions = cross_val_predict(model, X_train, y_train, cv=nfold)

    # define the grid search parameters
    batch_size = [100]
    epochs = [10]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=cpus,
                        cv=nfold)

    t0 = time()
    cv_score_ann = 0
    best_param_ann = []
    ann_model_file = os.path.join(outfolder, "ann-%s.m" % task)
    nfold_predictions = None

    if load_model:
        print("model is loaded from [%s]" % str(ann_model_file))
        best_estimator = util.load_classifier_model(ann_model_file)
    else:
        grid.fit(text_based_features, y_train)
        nfold_predictions = cross_val_predict(grid.best_estimator_, text_based_features, y_train, cv=nfold)

        cv_score_ann = grid.best_score_
        best_param_ann = grid.best_params_
        best_estimator = grid.best_estimator_

        # self.save_classifier_model(best_estimator, ann_model_file)

    print(datetime.datetime.now())
    print("testing on development set ....")
    if (X_test is not None):
        heldout_predictions_final = best_estimator.predict(X_test)
        util.save_scores(nfold_predictions, y_train, heldout_predictions_final, y_test, model, task, identifier, 2,
                         outfolder)

    else:
        util.save_scores(nfold_predictions, y_train, None,y_test,"dnn", task, identifier, 2, outfolder)

    # util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
    #                       time_ann_predict_dev,
    #                       time_ann_train, y_test)


def create_model(model_descriptor: str, max_index=100, wemb_matrix=None, append_feature_matrix=None):
    '''A model that uses word embeddings'''
    word_embedding_dim_output = 300
    max_sequence_length_profile = 100
    if wemb_matrix is None:
        if append_feature_matrix is not None:
            embedding_layers = [Embedding(input_dim=max_index, output_dim=word_embedding_dim_output,
                                          input_length=max_sequence_length_profile),
                                Embedding(input_dim=max_index, output_dim=len(append_feature_matrix[0]),
                                          weights=[append_feature_matrix],
                                          input_length=max_sequence_length_profile,
                                          trainable=False)]
        else:
            embedding_layers = [Embedding(input_dim=max_index, output_dim=word_embedding_dim_output,
                                          input_length=max_sequence_length_profile)]

    else:
        if append_feature_matrix is not None:
            concat_matrices = du.concat_matrices(wemb_matrix, append_feature_matrix)
            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layers = [Embedding(input_dim=max_index, output_dim=len(concat_matrices[0]),
                                          weights=[concat_matrices],
                                          input_length=max_sequence_length_profile,
                                          trainable=False)]
        else:
            embedding_layers = [Embedding(input_dim=max_index, output_dim=len(wemb_matrix[0]),
                                          weights=[wemb_matrix],
                                          input_length=max_sequence_length_profile,
                                          trainable=False)]

    #if model_descriptor.startswith("b_"):
    model_descriptor = model_descriptor[2:].strip()
    model = du.create_model_with_branch(embedding_layers, model_descriptor)

    #model = du.create_final_model_with_concat_cnn(embedding_layers, model_descriptor)

    # create_model_conv_lstm_multi_filter(embedding_layer)

    # logger.info("New run started at {}\n{}".format(datetime.datetime.now(), model.summary()))
    return model
