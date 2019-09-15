import sys
import os
import datetime

import numpy
from numpy.random import seed

from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd

seed(1)
os.environ['PYTHONHASHSEED'] = '0'

def create_features(csv_feature_folder):
    csv_features = csv_feature_folder + "/features.csv"

    label_col=15

    df = pd.read_csv(csv_features, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()
    y = df[:, label_col]

    X = df[:, 1:15]
    # Convert feature vectors to float64 type
    X = X.astype(numpy.float32)

    return X, y

if __name__ == "__main__":
    #this is the file pointing to the basic features, i.e., just the numeric values
    #msm4phi/paper2/data/training_data/basic_features.csv
    csv_feature_folder=sys.argv[1]

    #this is the folder to save output to
    outfolder=sys.argv[2]
    n_fold=10

    print(datetime.datetime.now())
    X, y=create_features(csv_feature_folder)

    #behaviour only
    print(">>>>> _uddin2018_ >>>>>")
    print(datetime.datetime.now())
    cls = cm.Classifer("stakeholdercls", "_uddin18_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["svm"])
    cls.run()


