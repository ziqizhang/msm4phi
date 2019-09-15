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
    csv_feature_other = csv_feature_folder + "/features_others.csv"
    csv_feature_cos = csv_feature_folder + "/features_ot_rt_cos.csv"

    label_col=62

    df_other = pd.read_csv(csv_feature_other, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()
    df_cos = pd.read_csv(csv_feature_cos, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()

    cos_features={}
    for r in df_cos:
        user=r[0]
        features=r[1:]
        cos_features[user]=features

    X=numpy.zeros((len(df_other),65))
    y=[]
    index=0
    for r in df_other:
        user=r[0]
        label=r[label_col]
        features=r[1:label_col]
        features_cos=cos_features[user]
        features=numpy.hstack([features,features_cos])
        X[index]=features
        y.append(label)
        index+=1

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
    print(">>>>> _kim2014_ >>>>>")
    print(datetime.datetime.now())
    cls = cm.Classifer("stakeholdercls", "_kim2014_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["gbrt"])
    cls.run()


