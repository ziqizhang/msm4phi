import sys

from exp import feature_creator as fc
from classifier import classifier_main as cm

if __name__ == "__main__":
    #this is the file pointing to the basic features, i.e., just the numeric values
    #msm4phi/paper2/data/training_data/basic_features.csv
    csv_basic_feature=sys.argv[1]
    #this is the folder containing other extracted features
    csv_other_feature=sys.argv[2]
    #this is the folder to save output to
    outfolder=sys.argv[3]

    #SETTING1 basic features
    X, y=fc.create_basic(csv_basic_feature)
    cls = cm.Classifer("stakeholdercls","basic", X, y,outfolder)
    cls.run()


    #SETTING2 basic features + tweet_feature/diseases_in_tweets.csv
    X,y=fc.create_basic_and_autocreated_dictionary(csv_basic_feature,csv_other_feature)
    cls = cm.Classifer("stakeholdercls", "basic+tweetfeature1", X, y, outfolder)
    cls.run()