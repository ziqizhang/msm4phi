import csv
import urllib.request
import pandas as pd

import sys


def optimize_solr_index(solr_url, corename):
    code = urllib.request. \
        urlopen("{}/{}/update?optimize=true".
                format(solr_url, corename)).read()

def merge_csv_files(csv1, csv1_id_col, csv1_label_col, csv2, csv2_id_col, outfile):
    df = pd.read_csv(csv2, header=None, delimiter=",", quoting=0
                     ).as_matrix()
    csv2_dict = dict()
    for row in df:
        row=list(row)
        id  =row[csv2_id_col]
        del row[csv2_id_col]
        csv2_dict[id] = row

    df = pd.read_csv(csv1, header=None, delimiter=",", quoting=0
                     ).as_matrix()
    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in df:
            row=list(row)
            id=row[csv1_id_col]
            label=row[csv1_label_col].strip()
            del row[csv1_label_col]
            row=row+csv2_dict[id]+[label]
            csvwriter.writerow(row)


if __name__=="__main__":
    optimize_solr_index(sys.argv[1],sys.argv[2])
    # merge_csv_files("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/Georgica/classifier/output_features.csv",0, 14,
    #                 "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/user_features_2.csv",0,
    #                 "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/user_features_and_labels_2.csv")

