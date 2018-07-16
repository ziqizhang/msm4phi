import sys

import datetime
from numpy.random import seed

from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd
seed(1)

if __name__ == "__main__":
    #this is the file pointing to the basic features, i.e., just the numeric values
    #msm4phi/paper2/data/training_data/basic_features.csv
    csv_basic_feature=sys.argv[1]
    #this is the folder containing other extracted features
    csv_other_feature=sys.argv[2]
    #this is needed if dnn model is used
    dnn_embedding_file="/home/zz/Work/data/glove.840B.300d.bin.gensim"
    #dnn_embedding_file = "/home/zz/Work/data/Set1_TweetDataWithoutSpam_Word.bin"

    #this is the folder to save output to
    outfolder=sys.argv[3]

    # SETTING1 dnn applied to profile, with numeric features
    print(datetime.datetime.now())
    X, y = fc.create_basic(csv_basic_feature)
    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    df.astype(str)
    profiles = df[:, 22]
    profiles = ["" if type(x) is float else x for x in profiles]
    cls = cm.Classifer("stakeholdercls", "_dnn_text+basic_", X, y, outfolder,
                       text_data=profiles, dnn_embedding_file=dnn_embedding_file)
    cls.run()


    print(datetime.datetime.now())
    X, y = fc.create_numeric(csv_basic_feature, csv_other_feature)
    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    df.astype(str)
    profiles = df[:, 22]
    profiles = ["" if type(x) is float else x for x in profiles]
    cls = cm.Classifer("stakeholdercls", "_dnn_text+numeric_", X, y, outfolder,
                       text_data=profiles, dnn_embedding_file=dnn_embedding_file)
    cls.run()

    print(datetime.datetime.now())
    X, y = fc.create_autocreated_dictext(csv_basic_feature, csv_other_feature)
    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    df.astype(str)
    profiles = df[:, 22]
    profiles = ["" if type(x) is float else x for x in profiles]
    cls = cm.Classifer("stakeholdercls", "_dnn_text+autodictext_", X, y, outfolder,
                       text_data=profiles, dnn_embedding_file=dnn_embedding_file)
    cls.run()

    print(datetime.datetime.now())
    X, y = fc.create_text_and_numeric_and_autodictext(csv_basic_feature, csv_other_feature)
    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    df.astype(str)
    profiles = df[:, 22]
    profiles = ["" if type(x) is float else x for x in profiles]
    cls = cm.Classifer("stakeholdercls", "_dnn_text+numeric+autodictext_", X, y, outfolder,
                       text_data=profiles, dnn_embedding_file=dnn_embedding_file)
    cls.run()
