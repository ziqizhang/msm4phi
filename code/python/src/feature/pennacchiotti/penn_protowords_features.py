# Given the original user_features.csv file that is a simple dump of indexed user features,
# this code processes it further to extract different kinds of features based on dictionaries
import collections
import csv
import re

import numpy
import pandas as pd
from feature import dictionary_extractor as de
from feature import nlp


# input: a dictionary containing different dictionaries to be used
def match_extracted_dictionary(dictionaries: dict, csv_input_feature_file, col_id, outfile,
                               *col_target_texts):
    dict_labels=list(dictionaries.keys())

    df = pd.read_csv(csv_input_feature_file, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()

    all_dictionry_entries=set()
    for d in dictionaries.values():
        all_dictionry_entries.update(d)
    all_dictionry_entries=list(all_dictionry_entries)

    output_matrix = []


    output_header = ["user_id"]
    for i in range(len(all_dictionry_entries)+len(dictionaries)):
        output_header.append(0)
    output_matrix.append(output_header)

    count = 0
    for row in df:
        print(count)
        count += 1
        row_data = [row[col_id]]
        target_text = ""
        features_proto_wp=numpy.zeros(len(all_dictionry_entries))

        skip = False
        for tt_col in col_target_texts:
            text = row[tt_col]
            if type(text) is float:
                skip = True
                break

            target_text += text + " "

        if skip:
            row_data.append(list(features_proto_wp))
            output_matrix.append(row_data)
            continue

        target_text = target_text.strip()


        features_proto_c=[]
        for k in dict_labels:
            dictionary = dictionaries[k]
            matchsum, total_words, match_freq = \
                count_word_matches(dictionary, target_text, de.text_normalization_option)
            if total_words==0:
                total_words=1
            features_proto_c.append(matchsum/total_words)
            for i in range(len(all_dictionry_entries)):
                entry=all_dictionry_entries[i]
                if entry in match_freq.keys():
                    freq=match_freq[entry]
                    try:
                        features_proto_wp[i]=features_proto_wp[i]+(freq/total_words)
                    except IndexError:
                        print()

        row_data.extend(list(features_proto_wp))
        row_data.extend(features_proto_c)
        output_matrix.append(row_data)

    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_matrix:
            csvwriter.writerow(row)


def count_word_matches(dictionary, target_text, text_normalization_option):
    target_text = nlp.normalize_tweet(target_text)
    norm_toks = nlp.tokenize(target_text, text_normalization_option)
    freq=collections.Counter(norm_toks)
    matchsum = 0

    matched_freq={}
    for w, f in freq.items():
        if w in dictionary.keys():
            matchsum += f
            matched_freq[w]=f

    return matchsum, len(norm_toks), matched_freq


# Profile: pattern, e.g., 'VERB X X HealthCondition'
def flatten_dictionary(postype_dictionaries):
    out_dict = {}
    for postype, dictionaries in postype_dictionaries.items():
        for label, dicts in dictionaries.items():
            if label == "Other":
                continue
            out_dict[postype + "_" + label] = dicts
    return out_dict


def load_generic_dictionary(txtfile):
    with open(txtfile) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip().lower() for x in content]
    return content


if __name__ == "__main__":
    # root folder containing the dictionaries
    dictionary_folder = "/home/zz/Work/msm4phi_data/paper2/reported/penn/dictionary_word"
    outfolder = "/home/zz/Work/msm4phi_data/paper2/reported/penn"
    #this is the training data file containing profile texts
    csv_feature_input = "/home/zz/Work/msm4phi_data/paper2/reported/training_data/paper_reported" \
                        "/basic_features_empty_profile_filled(tweets).csv"

    # column id of the target text field
    target_text_cols = 22  # 22=profile text; 15=name field
    target_text_name_suffix = "_profile"
    col_id = 14
    # how many entries from each dictionary should be selected for matching(top n). Changing this param will generate
    # different features, so perhaps influencing classification results
    topN_of_dict = 100

    file = csv_feature_input
    # load auto extracted dictionaries, match to 'profile' (make sure this matches to csv_feature_input)
    # i.e., if dictionary were extracted from bio, then the csv_feature_input needs to match bio, otherwise...
    postype_dictionaries = \
        de.load_extracted_dictionary(
            dictionary_folder, #this is Penn's method
            topN_of_dict, "any") #any-any words regardless of classes
    extracted_dictionaries = flatten_dictionary(postype_dictionaries)

    match_extracted_dictionary(extracted_dictionaries, file,
                               col_id,
                               outfolder + "/feature_autocreated_dict_match" + target_text_name_suffix + ".csv",
                               target_text_cols)
