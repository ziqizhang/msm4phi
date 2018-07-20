from classifier import classifier_util as util
import os

'''this file loads a pre-trained model and classify new data'''

def predict(model_flag, task, test_data, outfolder):
    print("start prediction stage :: data size:", len(test_data))

    ######################### SGDClassifier #######################
    model_file=None
    if model_flag== "sgd":
        # SGD doesn't work so well with only a few samples, but is (much more) performant with larger data
        # At n_iter=1000, SGD should converge on most datasets
        print("Using SGD ...")
        model_file = os.path.join(outfolder, "sgd-classifier-%s.m" % task)

    ######################### Stochastic Logistic Regression#######################
    if model_flag== "lr":
        print("Using Stochastic Logistic Regression ...")
        model_file = os.path.join(outfolder, "stochasticLR-%s.m" % task)

    ######################### Random Forest Classifier #######################
    if model_flag== "rf":
        print("Using Random Forest ...")
        model_file = os.path.join(outfolder, "random-forest_classifier-%s.m" % task)

    ###################  liblinear SVM ##############################
    if model_flag== "svm_l":
        print("Using SVM, kernel=linear ...")
        model_file = os.path.join(outfolder, "liblinear-svm-linear-%s.m" % task)

    ##################### RBF svm #####################
    if model_flag== "svm_rbf":
        print("Using SVM, kernel=rbf ....")
        model_file = os.path.join(outfolder, "liblinear-svm-rbf-%s.m" % task)

    model = util.load_classifier_model(model_file)
    predictions = model.predict_proba(test_data)
    util.saveOutput(predictions, model_flag, task, outfolder)
