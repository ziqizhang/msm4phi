import sys

import datetime

from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd

if __name__ == "__main__":
    #this is the file pointing to the basic features, i.e., just the numeric values
    #msm4phi/paper2/data/training_data/basic_features.csv
    csv_basic_feature_folder=sys.argv[1]
    #this is the folder containing other extracted features
    csv_other_feature_folder=sys.argv[2]
    #this is needed if dnn model is used
    dnn_embedding_file="/home/zz/Work/data/glove.840B.300d.bin.gensim"

    #this is the folder to save output to
    outfolder=sys.argv[3]
    n_fold=10


    datafeatures={}
    # datafeatures["full_"]=(csv_basic_feature_folder+"/basic_features.csv",
    #                        csv_other_feature_folder+"/full")
    # datafeatures["emtpyremoved_"] = (csv_basic_feature_folder+"/basic_features_empty_profile_removed.csv",
    #                        csv_other_feature_folder+"/empty_profile_removed")
    datafeatures["emptyfilled_"] = (csv_basic_feature_folder+"/basic_features_empty_profile_filled.csv",
                           csv_other_feature_folder+"/empty_profile_filled")

    ######## no pca #######
    for k,v in datafeatures.items():
        print(datetime.datetime.now())
        csv_text_and_behaviour=v[0]
        csv_preprocessed_feature=v[1]

        # behaviour only
        print(">>>>> _behaviour_only_ >>>>>")
        print(datetime.datetime.now())
        X, y = fc.create_behaviour(csv_text_and_behaviour, True)
        cls = cm.Classifer(k + "stakeholdercls", "_behaviour_only_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["sgd", "svm_l", "lr", "rf", "svm_rbf",
                                                                            "pca-sgd","pca-svm_l", "pca-lr", "pca-rf", "pca-svm_rbf"])
        cls.run()

        # dict only
        print(">>>>> _autodictext_only_ >>>>>")
        print(datetime.datetime.now())
        X, y = fc.create_autodict(csv_text_and_behaviour, csv_preprocessed_feature)
        cls = cm.Classifer(k + "stakeholdercls", "_autodictext_only_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["sgd", "svm_l", "lr", "rf", "svm_rbf",
                                                                            "pca-sgd", "pca-svm_l", "pca-lr", "pca-rf",
                                                                            "pca-svm_rbf"])
        cls.run()

        #text only
        print(">>>>> _text_only_ >>>>>")
        X, y = fc.create_textprofile(csv_text_and_behaviour)
        cls = cm.Classifer(k+"stakeholdercls", "_text_only_", X, y, outfolder,
                             categorical_targets=6,nfold=n_fold,algorithms=["sgd","svm_l","lr","rf","svm_rbf",
                                                                            "pca-sgd", "pca-svm_l", "pca-lr", "pca-rf",
                                                                            "pca-svm_rbf"])
        cls.run()


        #text+behaviour
        print(">>>>> _text+behaviour_only_ >>>>>")
        print(datetime.datetime.now())
        X, y = fc.create_text_and_behaviour(csv_text_and_behaviour, csv_preprocessed_feature)
        cls = cm.Classifer(k+"stakeholdercls", "_text+behaviour_", X, y, outfolder,
                            categorical_targets=6,nfold=n_fold,algorithms=["sgd","svm_l","lr","rf","svm_rbf",
                                                                           "pca-sgd", "pca-svm_l", "pca-lr", "pca-rf",
                                                                           "pca-svm_rbf"])
        cls.run()

        #text+dict
        print(">>>>> _text+dict_only_ >>>>>")
        print(datetime.datetime.now())
        X, y = fc.create_text_and_autodict(csv_text_and_behaviour, csv_preprocessed_feature)
        df = pd.read_csv(csv_text_and_behaviour, header=0, delimiter=",", quoting=0).as_matrix()
        cls = cm.Classifer(k + "stakeholdercls", "_text+dict_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["sgd", "svm_l", "lr", "rf", "svm_rbf",
                                                                            "pca-sgd", "pca-svm_l", "pca-lr", "pca-rf",
                                                                            "pca-svm_rbf"])
        cls.run()

        #text+behaviour+dict
        print(">>>>> _text+dict+behaviour_only_ >>>>>")
        print(datetime.datetime.now())
        X, y = fc.create_text_and_autodict_and_behaviour(csv_text_and_behaviour, csv_preprocessed_feature)
        df = pd.read_csv(csv_text_and_behaviour, header=0, delimiter=",", quoting=0).as_matrix()
        cls = cm.Classifer(k + "stakeholdercls", "_text+behaviour+dict_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["sgd", "svm_l", "lr", "rf", "svm_rbf",
                                                                            "pca-sgd", "pca-svm_l", "pca-lr", "pca-rf",
                                                                            "pca-svm_rbf"])
        cls.run()


        # cls.run()
        #
        # # Setting 6 text+numeric
        # print(datetime.datetime.now())
        # X, y = fc.create_text_and_numeric(csv_basic_feature, csv_other_feature)
        # cls = cm.Classifer(k+"stakeholdercls", "_text+numeric_", X, y, outfolder,
        #                    categorical_targets=6,nfold=n_fold,algorithms=["svm_l"])
        # cls.run()
        #
        # Setting 9 text+autocreated_dictext
        # print(datetime.datetime.now())
        # X, y = fc.create_text_and_autodictext(csv_basic_feature, csv_other_feature)
        # cls = cm.Classifer(k+"stakeholdercls", "_text+autodictext_", X, y, outfolder,
        #                    categorical_targets=6,nfold=n_fold,algorithms=["svm_l"])
        # cls.run()

        # setting 10 text+autodict_ext+numeric
        # print(datetime.datetime.now())
        # X, y = fc.create_text_and_numeric_and_autodictext(csv_basic_feature, csv_other_feature)
        # cls = cm.Classifer(k+"stakeholdercls", "_text+numeric+autodictext_", X, y, outfolder,
        #                    categorical_targets=6,nfold=n_fold,algorithms=["svm_l"])
        # cls.run()
        #


        ######## svm, pca #######

        # print(datetime.datetime.now())
        # X, y = fc.create_textprofile(csv_text_and_behaviour)
        # cls = cm.Classifer(k+"stakeholdercls", "_text_only_", X, y, outfolder,
        #                    categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        # cls.run()
        #
        # # Setting 6 text+numeric
        # print(datetime.datetime.now())
        # X, y = fc.create_text_and_numeric(csv_basic_feature, csv_other_feature)
        # cls = cm.Classifer(k+"stakeholdercls", "_text+numeric_", X, y, outfolder,
        #                    categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        # cls.run()
        #
        # Setting 9 text+autocreated_dictext
        # print(datetime.datetime.now())
        # X, y = fc.create_text_and_autodictext(csv_basic_feature, csv_other_feature)
        # cls = cm.Classifer(k+"stakeholdercls", "_text+autodictext_", X, y, outfolder,
        #                    categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        # cls.run()

        # setting 10 text+autodict_ext+numeric
        # print(datetime.datetime.now())
        # X, y = fc.create_text_and_numeric_and_autodictext(csv_basic_feature, csv_other_feature)
        # cls = cm.Classifer(k+"stakeholdercls", "_text+numeric+autodictext_", X, y, outfolder,
        #                    categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        # cls.run()
        #
        # # SETTING5 autocreated_dict extended only
        # print(datetime.datetime.now())
        # X, y = fc.create_autocreated_dictext(csv_basic_feature, csv_other_feature)
        # cls = cm.Classifer(k+"stakeholdercls", "_autodictext_only_", X, y, outfolder,
        #                    categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        # cls.run()


