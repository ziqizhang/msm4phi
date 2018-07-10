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

    #SETTING2 basic features - english profiles
    ###### file needs to be changed ##########
    print(datetime.datetime.now())
    X, y=fc.create_basic(csv_basic_feature)
    cls = cm.Classifer("stakeholdercls","basic - english", X, y,outfolder)
    cls.run()
    #########################################


    #SETTING3 basic features + tweet_feature/diseases_in_tweets.csv
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_diseases_in_tweets(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+diseases_in_tweets", X, y, outfolder)
    cls.run()

    #SETTING4 basic features + tweet_feature/topical_tweets.csv
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_topical_tweets(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+topical_tweets", X, y, outfolder)
    cls.run()

    #SETTING2 basic features + tweet_feature/diseases_in...csv + tweet_feature/topical_tweets.csv
    # print(datetime.datetime.now())
    # X,y=fc.create_basic_diseases_and_topical_tweets(csv_basic_feature,csv_other_feature)
    # cls = cm.Classifer("stakeholdercls", "basic+diseases+topical_tweets", X, y, outfolder)
    # cls.run()


    #SETTING5 basic features + manual_dictionary_1
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_manual_dictionary(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+manual_dictionary", X, y, outfolder)
    cls.run()

    #SETTING6 basic_features + manual_dict_georgica
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_manual_dictionary_g(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+manual_dictionary_g", X, y, outfolder)
    cls.run()


    #SETTING7 basic features + dictionary_feature1/ autocreated_dict_match_profile.csv
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_autocreated_dictionary(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+autocreated_dict", X, y, outfolder)
    cls.run()
    ##################################################################################
    #SETTING8 basic features + dictionary_feature1/ disease_hashtag_match_profile.csv
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_hashtag_match_profile(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+hashtag_match_profile", X, y, outfolder)
    cls.run()

    #SETTING9 basic features + dictionary_feature1/ disease_word_match_profile.csv
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_word_match_profile(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+word_match_profile", X, y, outfolder)
    cls.run()

    #SETTING10 basic features + dictionary_feature1/ generic_dict_match_name.csv
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_generic_dict_match_name(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+generic_dict_match_name", X, y, outfolder)
    cls.run()

    #SETTING11 basic features + dictionary_feature1/ generic_dict_match_profile.csv
    print(datetime.datetime.now())
    X,y=fc.create_basic_and_generic_dict_match_profile(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+generic_dict_match_profile", X, y, outfolder)
    cls.run()

    #############################################################################


    #SETTING12 manual_dictionary only
    print(datetime.datetime.now())
    X,y=fc.create_manual_dict(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "manual_dict_only", X, y, outfolder)
    cls.run()

    #SETTING13 autocreated_dict only
    print(datetime.datetime.now())
    X,y=fc.create_autocreated_dict(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "autocreated_dict_only", X, y, outfolder)
    cls.run()




