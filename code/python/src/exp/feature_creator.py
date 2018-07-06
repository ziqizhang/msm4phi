import numpy
import pandas as pd

#this is the basic feature loader, using only the stats from indexed data.
def create_basic(csv_basic_feature):
    feature_start_col=1
    feature_end_col=13
    label_col=40

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    y=df[:,label_col]

    X = df[:, feature_start_col:feature_end_col+1]
    # Convert feature vectors to float64 type
    X = X.astype(numpy.float32)


    return X,y


def create_basic_and_autocreated_dictionary(csv_basic_feature, folder_other):
    X, y=create_basic(csv_basic_feature)
    csv_autocreated_dict_feature=folder_other+\
        "/tweet_feature/diseases_in_tweets.csv"
    df = pd.read_csv(csv_autocreated_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:,2:]
    X_2 = X_2.astype(numpy.float32)
    X_new=numpy.concatenate([X, X_2],axis=1) #you can keep concatenating other matrices here. but
    #remember to call 'astype' like above on them before concatenating
    return X_new, y


