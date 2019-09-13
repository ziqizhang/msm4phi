import sys
from exp import annotator as ann
import os


import datetime
from numpy.random import seed

seed(1)
os.environ['PYTHONHASHSEED'] = '0'

from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd


#DNN_MODEL_DESCRIPTOR= "cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"
#DNN_MODEL_DESCRIPTOR="lstm=100-False|dense=6-softmax|glv"
#DNN_MODEL_DESCRIPTOR="bilstm=100-False|dense=6-softmax|glv"
#DNN_MODEL_DESCRIPTOR="scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=6-softmax|glv"
#DNN_MODEL_DESCRIPTOR="scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"



if __name__ == "__main__":
    # this is the file pointing to the basic features, i.e., just the numeric values
    # msm4phi/paper2/data/training_data/basic_features.csv
    csv_basic_feature_folder = sys.argv[1]
    # this is the folder containing other extracted features
    csv_other_feature_folder = sys.argv[2]
    # this is needed if dnn model is used
    dnn_embedding_file = "/home/zz/Work/data/glove.840B.300d.bin.gensim"

    # this is the folder to save output to
    outfolder = sys.argv[3]

    tweets_exta=None
    if len(sys.argv)>4:
        tweets_exta=ann.generate_extra_data_for_embeddingvocab(sys.argv[4],sys.argv[5])

    n_fold = 10

    datafeatures = {}
    #datafeatures["full_"] = (csv_basic_feature_folder + "/basic_features.csv",
    #                         csv_other_feature_folder + "/full")
    # datafeatures["emtpyremoved_"] = (csv_basic_feature_folder + "/basic_features_empty_profile_removed.csv",
    #                                  csv_other_feature_folder + "/empty_profile_removed")
    datafeatures["emptyfilled_"] = (csv_basic_feature_folder + "/basic_features_empty_profile_filled.csv",
                                    # basic_features_empty_profile_filled - for features with profiles
                                    # basic_features_empty_profile_filled(tweets) - for features with concatenated tweets
                                    csv_other_feature_folder + "/empty_profile_filled")
                                # baseline_empty_profile_filled(tweets) - for dict features generated on tweets
                                # empty_profile_filled - for dict features generated on profiles


    model_descriptors = ["scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"]#,
                        # "bilstm=100-False"]

    for k, v in datafeatures.items():
        print(datetime.datetime.now())

        csv_text = v[0]
        csv_other_feature = v[1]
        print(csv_text)

        for model_descriptor in model_descriptors:
            print("\t"+model_descriptor)

            # print(">>>>>> behaviour only >>>")
            # print(datetime.datetime.now())
            # X, y = fc.create_behaviour(csv_text)
            # cls = cm.Classifer("stakeholdercls", "_dnn_behaviour_", X, y, outfolder,
            #                    categorical_targets=6, algorithms=["dnn"], nfold=n_fold,
            #                    text_data=None, dnn_embedding_file=None,
            #                    dnn_descriptor=model_descriptor)
            # cls.run()
            #
            # print(">>>>>> dict only >>>")
            # print(datetime.datetime.now())
            # X, y = fc.create_autodict(csv_text, csv_other_feature)
            # cls = cm.Classifer("stakeholdercls", "_dnn_dict_", X, y, outfolder,
            #                    categorical_targets=6, algorithms=["dnn"], nfold=n_fold,
            #                    text_data=None, dnn_embedding_file=None,
            #                    dnn_descriptor=model_descriptor)
            # cls.run()
            # #
            # #
            # # #SETTING0 dnn applied to profile only
            # print(">>>>>> text only >>>")
            # print(datetime.datetime.now())
            # X, y = fc.create_behaviour(csv_text)
            # df = pd.read_csv(csv_text, header=0, delimiter=",", quoting=0).as_matrix()
            # df.astype(str)
            # profiles = df[:, 22]
            # profiles = ["" if type(x) is float else x for x in profiles]
            # cls = cm.Classifer("stakeholdercls", "_dnn_text_", None, y, outfolder,
            #                    categorical_targets=6, algorithms=["dnn"], nfold=n_fold,
            #                    text_data=profiles, dnn_embedding_file=dnn_embedding_file,
            #                    dnn_descriptor=model_descriptor)
            # cls.run()
            #
            # # SETTING0 dnn applied to profile and stat meta feature only
            # print(">>>>>> text+behaviour only >>>")
            # print(datetime.datetime.now())
            # X, y = fc.create_behaviour(csv_text)
            # df = pd.read_csv(csv_text, header=0, delimiter=",", quoting=0).as_matrix()
            # df.astype(str)
            # profiles = df[:, 22]
            # profiles = ["" if type(x) is float else x for x in profiles]
            # cls = cm.Classifer("stakeholdercls", "_dnn_text+behaviour_", X, y, outfolder,
            #                    categorical_targets=6, algorithms=["dnn"], nfold=n_fold,
            #                    text_data=profiles, dnn_embedding_file=dnn_embedding_file,
            #                    dnn_descriptor=model_descriptor)
            # cls.run()

            print(">>>>>> text+dict only >>>")
            X, y = fc.create_autodict(csv_text, csv_other_feature)
            df = pd.read_csv(csv_text, header=0, delimiter=",", quoting=0).as_matrix()
            df.astype(str)
            profiles = df[:, 22]
            profiles = ["" if type(x) is float else x for x in profiles]
            cls = cm.Classifer("stakeholdercls", "_dnn_text+autodictext_", X, y, outfolder,
                               categorical_targets=6, algorithms=["dnn"],nfold=n_fold,
                               text_data=profiles, dnn_embedding_file=dnn_embedding_file,
                               dnn_descriptor=model_descriptor,
                               dnn_text_data_extra_for_embedding_vcab=tweets_exta)
            cls.run()
            #
            # print(">>>>>> text+behaviour+dict only >>>")
            # X, y = fc.create_behaviour_and_autodict(csv_text, csv_other_feature)
            # df = pd.read_csv(csv_text, header=0, delimiter=",", quoting=0).as_matrix()
            # df.astype(str)
            # profiles = df[:, 22]
            # profiles = ["" if type(x) is float else x for x in profiles]
            # cls = cm.Classifer("stakeholdercls", "_dnn_text+behaviour+autodictext_", X, y, outfolder,
            #                    categorical_targets=6, algorithms=["dnn"],nfold=n_fold,
            #                    text_data=profiles, dnn_embedding_file=dnn_embedding_file,
            #                    dnn_descriptor=model_descriptor)
            # cls.run()
