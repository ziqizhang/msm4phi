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
    X, y = fc.create_textprofile(csv_basic_feature)
    cls = cm.Classifer("stakeholdercls", "_text_only_", X, y, outfolder)
    cls.run()


    #SETTING2 basic features + tweet_feature/diseases_in_tweets.csv
    # print(datetime.datetime.now())
    # X,y=fc.create_numeric(csv_basic_feature, csv_other_feature)
    # cls = cm.Classifer("stakeholdercls", "_numeric_only_", X, y, outfolder)
    # cls.run()


    #SETTING3 manual_dictionary only
    # print(datetime.datetime.now())
    # X,y=fc.create_manual_dict(csv_basic_feature,csv_other_feature)
    # cls = cm.Classifer("stakeholdercls", "_manualdict_only_", X, y, outfolder)
    # cls.run()
    #
    # #SETTING4 autocreated_dict only
    # print(datetime.datetime.now())
    # X,y=fc.create_autocreated_dictext(csv_basic_feature, csv_other_feature)
    # cls = cm.Classifer("stakeholdercls", "_autodict_only_", X, y, outfolder)
    # cls.run()

    # Setting 6 text+numeric
    print(datetime.datetime.now())
    X, y = fc.create_text_and_numeric(csv_basic_feature, csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "_text+numeric_", X, y, outfolder)
    cls.run()

    # Setting 9 text+autocreated_dictext
    print(datetime.datetime.now())
    X, y = fc.create_text_and_autodictext(csv_basic_feature, csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "_text+autodictext_", X, y, outfolder)
    cls.run()

    # SETTING15 basic + autocreated_dict + tfidf weighted nagram
    # print(datetime.datetime.now())
    # X,y=fc.create_basic_auto_dict_and_text(csv_basic_feature, csv_other_feature)
    # cls = cm.Classifer("stakeholdercls", "_basic + autodict+text(p+n)_", X, y, outfolder)
    # cls.run()

    # setting 10 text+autodict_ext+numeric
    print(datetime.datetime.now())
    X, y = fc.create_text_and_numeric_and_autodictext(csv_basic_feature, csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "_text+numeric+autodictext_", X, y, outfolder)
    cls.run()

    #SETTING5 autocreated_dict extended only
    print(datetime.datetime.now())
    X, y = fc.create_autocreated_dictext(csv_basic_feature, csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "_autodictext_only_", X, y, outfolder)
    cls.run()



    #Setting 7 text+manual_dictionary
    # print(datetime.datetime.now())
    # X, y = fc.create_text_and_manualdict(csv_basic_feature, csv_other_feature)
    # cls = cm.Classifer("stakeholdercls", "_text+manualdict_", X, y, outfolder)
    # cls.run()
    # #Setting 8 text+autocreated_dict
    # print(datetime.datetime.now())
    # X, y = fc.create_text_and_autodict(csv_basic_feature, csv_other_feature)
    # cls = cm.Classifer("stakeholdercls", "_text+autodict_", X, y, outfolder)
    # cls.run()
