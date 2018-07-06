import sys

from exp import feature_creator as fc

if __name__ == "__main__":
    #this is the file pointing to the basic features, i.e., just the numeric values
    csv_basic_feature=sys.argv[0]
    #this is the folder containing other extracted features
    csv_dictionary_feature=sys.argv[1]

    #basic features
    X, y=fc.create_basic(csv_basic_feature)
    #classify and save


    #basic features + dictionary
    fc.create_basic_and_dictionary(csv_basic_feature,csv_dictionary_feature)