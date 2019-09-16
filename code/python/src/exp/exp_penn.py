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

def create_features(csv_feature_folder, training_data_csv):
    df = pd.read_csv(training_data_csv, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()
    y = df[:, 40]

    allfeatures={}
    allfeatures_dim={}
    for fe in os.listdir(csv_feature_folder):
        if '.csv' not in fe:
            continue
        f_df=pd.read_csv(csv_feature_folder+"/"+fe, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()
        map={}
        for row in f_df:
            user=row[0].strip()
            features=row[1:]
            if 'profile+tweeting' in fe:
                features=features[:len(features)-1]

            map[user]=features
            allfeatures[fe]=map
            allfeatures_dim[fe]=len(features)

    featuretypes=list(allfeatures.keys())
    X=[]
    for row in df:
        user=row[14]
        user_feature=[]
        for ft in featuretypes:
            entries=allfeatures[ft]
            if user not in entries.keys():
                dim=allfeatures_dim[ft]
                append=list(numpy.zeros(dim))
                print("ft={}, user={}".format(ft, user))
            else:
                append=entries[user]
            append= [0 if numpy.math.isnan(x) else x for x in append]
            user_feature.extend(append)
        X.append(user_feature)

    X = numpy.array(X)

    return X, y

if __name__ == "__main__":
    #this is the file pointing to the basic features, i.e., just the numeric values
    #msm4phi/paper2/data/training_data/basic_features.csv
    csv_feature_folder=sys.argv[1]

    #this is the folder to save output to
    training_data_csv=sys.argv[2]
    outfolder=sys.argv[3]
    n_fold=10

    print(datetime.datetime.now())
    X, y=create_features(csv_feature_folder, training_data_csv)

    #behaviour only
    print(">>>>> _penn11_ >>>>>")
    print(datetime.datetime.now())
    cls = cm.Classifer("stakeholdercls", "_penn11_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["gbrt"])
    cls.run()


