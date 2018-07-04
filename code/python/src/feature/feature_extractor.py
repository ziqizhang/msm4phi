# Given the original user_features.csv file that is a simple dump of indexed user features,
# this code processes it further to extract different kinds of features
import csv
import re

import pandas as pd
from feature import dictionary_extractor as de
from feature import nlp


# input: a dictionary containing different dictionaries to be used
def match_extracted_dictionary(dictionaries: dict, csv_input_feature_file, col_id, outfile,
                               *col_target_texts):
    df = pd.read_csv(csv_input_feature_file, header=0, delimiter=",", quoting=0).as_matrix()

    output_matrix = []

    output_header = ["user_id"]
    dict_labels = list(dictionaries.keys())
    for k in dict_labels:
        output_header.append(k + "_scoresum")
        output_header.append(k + "_matchsum")
        output_header.append(k + "_matchmax")
        output_header.append(k + "_matchbool")

    for row in df:
        row_data = [row[col_id]]
        target_text = ""
        for tt_col in col_target_texts:
            target_text += row[tt_col] + " "
        target_text = target_text.strip()

        if len(target_text) < 2:
            output_matrix.append(row_data)
            continue

        for k in dict_labels:
            dictionary = dictionaries[k]
            scoresum, matchsum, matchmax, matchbool = \
                find_word_matches(dictionary, target_text, de.text_normalization_option)
            row_data.append(scoresum)
            row_data.append(matchsum)
            row_data.append(matchmax)
            row_data.append(matchbool)
        output_matrix.append(row_data)

    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_matrix:
            csvwriter.writerow(row)


def match_extracted_healthconditions(dictionary: dict, csv_input_feature_file, col_id, outfile,
                                     *col_target_texts):
    df = pd.read_csv(csv_input_feature_file, header=0, delimiter=",", quoting=0).as_matrix()

    output_matrix = []

    output_header = ["user_id"]
    output_header.append("has_hc")
    output_header.append("count_hc")

    for row in df:
        row_data = [row[col_id]]
        target_text = ""
        for tt_col in col_target_texts:
            target_text += row[tt_col] + " "
        target_text = target_text.strip()

        if len(target_text) < 2:
            output_matrix.append(row_data)
            continue

        count_hc = find_hc_matches(dictionary, target_text)

        has_hc = 0
        if count_hc > 0:
            has_hc = 1
        output_matrix.append(has_hc)
        output_matrix.append(count_hc)

    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_matrix:
            csvwriter.writerow(row)


# person name, profession, title; against name, profile etc.
def match_generic_gazetteer(dictionaries: dict, csv_input_feature_file, col_id, outfile,
                            *col_target_texts):
    df = pd.read_csv(csv_input_feature_file, header=0, delimiter=",", quoting=0).as_matrix()

    output_matrix = []

    output_header = ["user_id"]
    dict_labels = list(dictionaries.keys())
    for k in dict_labels:
        output_header.append(k + "_hasmatch")

    for row in df:
        row_data = [row[col_id]]
        target_text = ""
        for tt_col in col_target_texts:
            target_text += row[tt_col] + " "
        target_text = target_text.strip()

        if len(target_text) < 2:
            output_matrix.append(row_data)
            continue

        toks = set(target_text.split(" "))
        for k in dict_labels:
            dictionary = dictionaries[k]
            if len(toks.intersection(dictionary)) > 0:
                output_matrix.append("1")
            else:
                output_matrix.append("0")

        output_matrix.append(row_data)

    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_matrix:
            csvwriter.writerow(row)


def find_hc_matches(dictionary: dict, target_text):
    hashtag_regex = '#[\w\-]+'
    matches = re.findall(hashtag_regex, target_text)
    for m in matches:
        target_text += " " + m[1:]
    target_text = target_text.strip()

    toks = set(target_text.split(" "))

    hc = set()
    inter = toks.intersection(dictionary.keys())
    for t in inter:
        hc.update(dictionary[t])

    return len(hc)


def find_word_matches(dictionary, target_text, text_normalization_option):
    target_text = nlp.normalize_tweet(target_text)
    norm_toks = set(nlp.tokenize(target_text, text_normalization_option))

    scoresum = 0
    matchsum = 0
    matchmax = 0
    matchbool = 0
    for w, score in dictionary.items():
        if w in norm_toks:
            matchbool = 1
            matchsum += 1
            scoresum += score
            if matchmax < score:
                matchmax = score

    return scoresum, matchsum, matchmax, matchbool

# Profile: pattern, e.g., 'VERB X X HealthCondition'
