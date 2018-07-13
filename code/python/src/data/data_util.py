import csv
import urllib.request
import pandas as pd
import glob

def optimize_solr_index(solr_url, corename):
    code = urllib.request. \
        urlopen("{}/{}/update?optimize=true".
                format(solr_url, corename)).read()


def merge_csv_files(csv1, csv1_id_col, csv1_label_col, csv2, csv2_id_col, outfile):
    df = pd.read_csv(csv2, header=None, delimiter=",", quoting=0
                     ).as_matrix()
    csv2_dict = dict()
    for row in df:
        row = list(row)
        id = row[csv2_id_col]
        del row[csv2_id_col]
        csv2_dict[id] = row

    df = pd.read_csv(csv1, header=None, delimiter=",", quoting=0
                     ).as_matrix()
    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in df:
            row = list(row)
            id = row[csv1_id_col]
            label = row[csv1_label_col].strip()
            del row[csv1_label_col]
            row = row + csv2_dict[id] + [label]
            csvwriter.writerow(row)


def filter_empty_profiles(csv_basic_feature_file, profile_col, new_csv_file):
    csv_basic_features = pd.read_csv(csv_basic_feature_file, sep=',', encoding="utf-8")

    with open(new_csv_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = list(csv_basic_features.columns.values)

        csvwriter.writerow(header)
        csv_basic_features = csv_basic_features.as_matrix()
        for row in csv_basic_features:
            if type(row[profile_col]) is float:
                continue
            csvwriter.writerow(row)


# for empty profiles, take most recent N tweets as their profile
def fill_empty_profiles(csv_basic_feature_file,
                        csv_tweets_file,
                        profile_col,
                        recentN,
                        new_csv_file):
    csv_basic_features = pd.read_csv(csv_basic_feature_file, sep=',', quotechar='"',encoding="utf-8")
    with open(csv_tweets_file, newline="\n") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    csv_tweets = [x.strip() for x in content]

    tweets_lookup = {}
    for row in csv_tweets:
        elements=row.split(",")
        tweets_lookup[elements[0]] = elements

    with open(new_csv_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = list(csv_basic_features.columns.values)

        csvwriter.writerow(header)
        csv_basic_features = csv_basic_features.as_matrix()
        for row in csv_basic_features:
            id = row[0]
            if type(row[profile_col]) is float and id in tweets_lookup.keys():
                tweets = tweets_lookup[id]

                concat = 0
                profile = ""
                for i in range(2, len(tweets)):
                    concat += 1
                    tweet = tweets[i]
                    if type(tweet) is str and len(tweet) > 2:
                        profile += tweet + " "
                    if concat == recentN:
                        break
                row[profile_col] = profile.strip()
            csvwriter.writerow(row)

def filter_empty_profile_features(filtered_csv_basic_feature_file,
                                  target_feature_file_folder):
    df = pd.read_csv(filtered_csv_basic_feature_file, header=None, delimiter=",", quoting=0
                     ).as_matrix()
    ids = set()
    for row in df:
        row = list(row)
        ids.add(row[0])


    files = glob.glob(target_feature_file_folder + '/**/*.csv', recursive=True)
    for f in files:
        csv_data = pd.read_csv(f, sep=',', quotechar='"', encoding="utf-8")
        header = list(csv_data.columns.values)
        csv_data=csv_data.as_matrix()
        with open(f, 'w', newline='\n') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(header)
            for row in csv_data:
                id=row[0]
                if id in ids:
                    csvwriter.writerow(row)

if __name__ == "__main__":
    # optimize_solr_index(sys.argv[1],sys.argv[2])
    # merge_csv_files("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/Georgica/classifier/output_features.csv",0, 14,
    #                 "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/user_features_2.csv",0,
    #                 "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/user_features_and_labels_2.csv")
    # filter_empty_profiles(
    #     "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/training_data/basic_features.csv",
    #     22,
    #     "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/training_data/basic_features_no_empty_profiles.csv")

    # fill_empty_profiles("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/training_data/basic_features.csv",
    #                     "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/features/tweet_feature/collected_tweets.csv",
    #                     22,
    #                     20,
    #                     "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/training_data/basic_features_filled_profiles.csv")

    filter_empty_profile_features(
        "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/training_data/basic_features_no_empty_profiles.csv",
        "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/features/empty_profile_removed")