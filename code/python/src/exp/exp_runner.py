import sys

import datetime

from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd

if __name__ == "__main__":
    #this is the file pointing to the basic features, i.e., just the numeric values
    #msm4phi/paper2/data/training_data/basic_features.csv
    csv_basic_feature=sys.argv[1]
    #this is the folder containing other extracted features
    csv_other_feature=sys.argv[2]
    #this is needed if dnn model is used
    dnn_embedding_file="/home/zz/Work/data/glove.840B.300d.bin.gensim"

    #this is the folder to save output to
    outfolder=sys.argv[3]


    # SETTING1 tfidf weighted nagram (text only. see code how features are
    # concatenated. You can do the same for text+other numeric) features
    print(datetime.datetime.now())
    X, y = fc.create_textfeatures_profile_and_name(csv_basic_feature)
    cls = cm.Classifer("stakeholdercls", "ngram_from_profiles", X, y, outfolder)
    cls.run()

    #SETTING2 basic features
    print(datetime.datetime.now())
    X, y=fc.create_basic(csv_basic_feature)
    cls = cm.Classifer("stakeholdercls","basic", X, y,outfolder)
    cls.run()
    #
    #
    # #SETTING2 basic features + tweet_feature/diseases_in_tweets.csv
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_autocreated_dictionary(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+tweetfeature1", X, y, outfolder)
    cls.run()