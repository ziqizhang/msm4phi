'''this file loads a pretrained model and annotates profiles saved in a folder.
it then updates the solr index with the annotations'''
import csv
import sys

import numpy

from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd
import os

def create_features():
    #generate correspondingly other numeric and dictionary features
    pass


def annotate():
    #for each folder
        #for each basic_feature.csv
            #find matching other numeric and dictionary features
            #annotate these profiles
            #save to index
    pass


def create_autocreated_dictext(csv_basic_feature, folder_other):
    X, y = fc.create_basic(csv_basic_feature, contains_label=False)

    other_feature_prefix="/"+csv_basic_feature[csv_basic_feature.rfind("/")+1:
                         csv_basic_feature.rfind(".")]

    csv_autocreated_dict_feature = folder_other + \
                                   other_feature_prefix+"_feature_autocreated_dict_match_profile.csv"
    df = pd.read_csv(csv_autocreated_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)

    csv_tweet_hashtag = folder_other + \
                        other_feature_prefix+"_feature_disease_hashtag_match_profile.csv"
    df = pd.read_csv(csv_tweet_hashtag, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_hashtag = df[:, 1:]
    X_tweet_hashtag = X_tweet_hashtag.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_hashtag, False)

    csv_tweet_word = folder_other + \
                     other_feature_prefix+"_feature_disease_word_match_profile.csv"
    df = pd.read_csv(csv_tweet_word, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_word = df[:, 1:]
    X_tweet_word = X_tweet_word.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_word, False)

    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_2, X_tweet_hashtag, X_tweet_word], axis=1)

    return X_all, y

if __name__ == "__main__":
    #/home/zz/Work/msm4phi_data/paper2/all_user_empty_filled_features
    csv_basic_feature_folder = sys.argv[1]
    # this is the folder containing other extracted features
    #/home/zz/Work/msm4phi_data/paper2/all_user_empty_filled_autodictext_features
    csv_other_feature_folder = sys.argv[2]
    pretrained_model_file=sys.argv[3]
    # this is needed if dnn model is used

    for csv_basic_feature in os.listdir(csv_basic_feature_folder):
        csv_other_feature = csv_other_feature_folder
        print(csv_basic_feature)

        X, y = create_autocreated_dictext(csv_basic_feature_folder+"/"+csv_basic_feature,
                                          csv_other_feature)
        # this is the folder to save output to
        dfraw = pd.read_csv(csv_basic_feature_folder+"/"+csv_basic_feature,
                            header=0, delimiter=",", quoting=0)
        df=dfraw.as_matrix()
        df.astype(str)
        profiles = df[:, 16]
        profiles = ["" if type(x) is float else x for x in profiles]

        outfolder = sys.argv[4]
        cls = cm.Classifer("stakeholdercls", "_dnn_text+autodictext_", X, y, outfolder,
                           categorical_targets=6, algorithms=["dnn"], nfold=None,
                           text_data=profiles, dnn_embedding_file=None,
                           dnn_descriptor="cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv")
        labels=cls.predict(pretrained_model_file)

        headers=list(dfraw.columns.values)
        headers.append("label")
        with open(outfolder+"/"+csv_basic_feature, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(headers)
            for i in range(0, len(df)):
                row=list(df[i])
                row.append(labels[i])
                csvwriter.writerow(row)
        exit(0)
