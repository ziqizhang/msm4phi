# this code copies 'dictionary_extractor.py' in 'feature' but applies to hashtags. see Pennachiotti2011
import csv

import os
import re
from collections import Counter

import pandas as pd


# input is a csv file that must a user at each row, and text to be concatenated to form user profile at different columns
# output: a dictionary containing a bag of profile text (normalised) for each label
from feature import nlp


def load_user_profile_text(user_features_csv, *col_text):
    df = pd.read_csv(user_features_csv, header=0, delimiter=",", quotechar='"'
                     )

    # process each label at a time to create a corpus of profile text for each label
    # sort the dataframe
    df.sort_values(by=['label'], inplace=True)
    # set the index to be this and don't drop
    df.set_index(keys=['label'], drop=False, inplace=True)
    # get a list of labels
    labels = df['label'].unique().tolist()

    label_to_proftext = dict()

    for l in labels:
        df_part = df.loc[df.label == l]

        proftext = []
        for index, row in df_part.iterrows():
            id = row[0]

            tweet_text = ""
            for c in col_text:
                if type(row[c])==str:
                    tweet_text += row[c] + " "

            tweet_text=nlp.normalize_tweet(tweet_text)
            if len(tweet_text) > 1:
                proftext.append(tweet_text)
        label_to_proftext[l]=proftext

    return label_to_proftext


def extract_dict(label_to_proftext: dict):
    # frequency based score
    label_vocab_class_freq = dict()
    vocab_cross_class_freq = dict()
    doc_freq=dict()

    for label, profile_docs in label_to_proftext.items():
        print(label+","+str(len(profile_docs)))
        vocab_freq = dict()

        for t in profile_docs:
            toks = nlp.tokenize(t)
            freq = Counter(toks)
            for k,v in freq.items():
                if k in vocab_freq.keys():
                    vocab_freq[k]=vocab_freq[k]+v
                else:
                    vocab_freq[k]=v
                if k in doc_freq.keys():
                    doc_freq[k]=vocab_freq[k]+1
                else:
                    doc_freq[k]=1

        label_vocab_class_freq[label] = vocab_freq
        for e, frequency in vocab_freq.items():
            if frequency==0:
                continue
            if e in vocab_cross_class_freq.keys():
                vocab_cross_class_freq[e] += frequency
            else:
                vocab_cross_class_freq[e] = frequency

    # calculate weighted score
    label_vocab_to_weightedscore = dict()
    for label, vocab_freq in label_vocab_class_freq.items():
        vocab_score = dict()
        for e, frequency in vocab_freq.items():
            if e not in vocab_cross_class_freq.keys():
                continue
            totalfreq = vocab_cross_class_freq[e]
            df=doc_freq[e]
            s = frequency / totalfreq #equation 1 in the OIR paper

            if s==1.0 or df<2:
                continue
            vocab_score[e] = s
        label_vocab_to_weightedscore[label] = vocab_score

    return label_vocab_class_freq, label_vocab_to_weightedscore

#rank by frequency (equation 1 in oir paper)
def rank_pass_one(outfolder, vocab_with_score: dict,
                  topN):
    for l, vocab in vocab_with_score.items():

        any = []
        sorted_by_value = sorted(vocab.items(), reverse=True, key=lambda kv: kv[1])

        idx = 0
        while (True and idx<len(sorted_by_value)):
            entry = sorted_by_value[idx]
            value = entry[1]
            entry=entry[0]
            if len(any) < topN:
                any.append(entry + "," + str(value))

            if len(any) == topN:
                break

            idx+=1

        # save the created dictionaries for this label
        file = open(outfolder + "/" + l + "_" + str(topN) + "_any.csv", 'w')
        for a in any:
            file.write(a + "\n")


#load dictionaries created by either rank_pass_one, or two
def load_extracted_dictionary(folder, topN,*permitted_postypes):
    postype_dictionary={}
    for file in os.listdir(folder):
        path_elements=os.path.split(file)
        identifiers = path_elements[len(path_elements)-1].split("_")
        label=identifiers[0]
        if len(identifiers)==3:
            postype=identifiers[2]
        else:
            postype=identifiers[1]
        if postype.endswith(".csv") or postype.endswith(".txt"):
            postype=postype[0: len(postype)-4]

        if postype not in permitted_postypes:
            continue

        if postype in postype_dictionary.keys():
            label_dictionary =postype_dictionary[postype]
        else:
            label_dictionary ={}

        if label in label_dictionary.keys():
            dictionary = label_dictionary[label]
        else:
            dictionary={}

        with open(folder+"/"+file, newline='\n') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            count=0
            for row in csvreader:
                if len(row)==2:
                    dictionary[row[0]]=row[1]
                else:
                    dictionary[row[0]]=0.0
                count+=1
                if count==topN:
                    break

        label_dictionary[label]=dictionary
        postype_dictionary[postype]=label_dictionary

    return postype_dictionary

if __name__ == "__main__":
    #col 40-label; col 22-desc;
    print("extracting text profiles...")
    profile_text = load_user_profile_text("/home/zz/Work/msm4phi_data/paper2/reported/training_data/paper_reported"
                                          "/basic_features_empty_profile_filled(tweets).csv",
                                          22) #22-profile; 15-name)
    print("calculating weighted freq...")
    vocab_to_totalfreq, vocab_to_weightedscore= \
        extract_dict(profile_text)

    print("ranking...")
    rank_pass_one(
        "/home/zz/Work/msm4phi_data/paper2/reported/penn/dictionary_word",
        vocab_to_weightedscore,
        100)


