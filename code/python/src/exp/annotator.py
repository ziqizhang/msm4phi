'''this file loads a pretrained model and annotates profiles saved in a folder.
it then updates the solr index with the annotations'''
import sys
from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd

def create_features():
    #generate correspondingly other numeric and dictionary features
    pass


def annotate():
    #for each folder
        #for each basic_feature.csv
            #find matching other numeric and dictionary features
            #annotate these profiles
            #save to index
    pass

if __name__ == "__main__":
    csv_basic_feature_folder = sys.argv[1]
    # this is the folder containing other extracted features
    csv_other_feature_folder = sys.argv[2]
    pretrained_model_file=sys.argv[3]
    # this is needed if dnn model is used

    csv_basic_feature = csv_basic_feature_folder
    csv_other_feature = csv_other_feature_folder
    print(csv_basic_feature)

    X, y = fc.create_autocreated_dictext(csv_basic_feature, csv_other_feature)
    # this is the folder to save output to
    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    df.astype(str)
    profiles = df[:, 22]
    profiles = ["" if type(x) is float else x for x in profiles]

    outfolder = sys.argv[4]
    cls = cm.Classifer("stakeholdercls", "_dnn_text+autodictext_", X, y, outfolder,
                       categorical_targets=6, algorithms=["dnn"], nfold=None,
                       text_data=profiles, dnn_embedding_file=None,
                       dnn_descriptor="scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv")
    cls.predict(pretrained_model_file)