import numpy
import os
import sys
import csv

from datetime import datetime

import pandas
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_classify(texts:set, vader_sentiment:SentimentIntensityAnalyzer):
    sents={}
    for t in texts:
        scores=vader_sentiment.polarity_scores(t)
        compound=scores['compound']
        if compound>=0.05:
            sents[t]='pos'
        elif compound<=-0.05:
            sents[t]='neg'
        else:
            sents[t]='neu'

    return sents

def match_keywords(texts:set, keywords:list):
    matches={}

    for t in texts:
        t=t.lower()

        matched=[]
        for k in keywords:
            if k in t:
                matched.append(k)

        matches[t]=matched
    return matches

def populate_features(user:str, texts:list, text_sentiment:dict, matched_keywords:dict,
                      keyword_index:map):
    row=[user]

    feature_pos=numpy.zeros(len(keyword_index))
    feature_neg=numpy.zeros(len(keyword_index))
    feature_neu=numpy.zeros(len(keyword_index))

    for t in texts:
        polarity=text_sentiment[t]
        matches=matched_keywords[t]

        for m in matches:
            index=keyword_index[m] #find index of the matched keyword in tweet
            if polarity=='pos':
                feature_pos[index]=feature_pos[index]+1
            if polarity == 'neg':
                feature_neg[index] = feature_neg[index] + 1
            if polarity == 'neu':
                feature_neu[index] = feature_neu[index] + 1

    feature_pos = feature_pos/len(texts)
    feature_neg = feature_neg / len(texts)
    feature_neu = feature_neu / len(texts)

    mean_pos=numpy.mean(feature_pos)
    mean_neg=numpy.mean(feature_neg)
    mean_neu=numpy.mean(feature_neu)

    std_pos = numpy.std(feature_pos)
    std_neg = numpy.std(feature_neg)
    std_neu = numpy.std(feature_neu)

    count_pos=numpy.count_nonzero(feature_pos)
    count_neg = numpy.count_nonzero(feature_neg)
    count_neu = numpy.count_nonzero(feature_neu)

    row.extend(list(feature_pos))
    row.extend(list(feature_neg))
    row.extend(list(feature_neu))
    row.extend([mean_pos, mean_neg, mean_neu, std_pos,std_neg,std_neu, count_pos,count_neg,count_neu])
    return row

def load_user_tweets(in_folder):
    user_tweets={}
    for af in os.listdir(in_folder):
        f = open(in_folder + "/" + af, "r")
        lines = f.readlines()
        f.close()

        text = []
        user = lines[0].strip()
        for i in range(1, len(lines)):
            parts = lines[i].lower().split("|")
            if len(parts) < 2:
                continue
            l = parts[1].strip()
            if (l.startswith("rt ")):
                l = l[3:]
            text.append(l)
            user_tweets[user]=text
    return user_tweets

def load_keywords(in_file):
    keywords={}
    df = pandas.read_csv(in_file, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()

    count=0
    for r in df:
        k = r[0]
        keywords[k]=count
        count+=1

    return keywords

if __name__ == "__main__":
    in_user_tweets=sys.argv[1]
    in_keywords=sys.argv[2] #/home/zz/Work/msm4phi_data/paper2/reported/hashtag_keywords.csv
    out_features=sys.argv[3]

    print("Loading data, at {}".format(datetime.now()))
    analyser = SentimentIntensityAnalyzer()
    user_tweets=load_user_tweets(in_user_tweets)
    keywords_index= load_keywords(in_keywords)
    keywords=list(keywords_index.keys())

    with open(out_features, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header=["user"]
        feature_cols=len(keywords)*3+9
        for i in range(feature_cols):
            header.append(i)
        csvwriter.writerow(header)

        for k, v in user_tweets.items():
            user=k
            tweets=v
            print("for user="+user)
            polarities=sentiment_classify(tweets, analyser)
            matched=match_keywords(tweets, keywords)
            row=populate_features(user, tweets, polarities, matched, keywords_index)
            csvwriter.writerow(row)