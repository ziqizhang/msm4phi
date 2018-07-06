import numpy
import pandas as pd

#this is the basic feature loader, using only the stats from indexed data.
def create_basic(csv_basic_feature):
    feature_start_col=1
    feature_end_col=15
    label_col=40

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0,
                     usecols=range(feature_start_col,feature_end_col)).as_matrix()
    y=df[label_col]

    X = df[:, feature_start_col:feature_end_col]
    # Convert feature vectors to float64 type
    X = X.astype(numpy.float64)


    return X,y


def create_basic_and_dictionary(csv_basic_feature, folder_other):
    X, y=create_basic(csv_basic_feature)

    

