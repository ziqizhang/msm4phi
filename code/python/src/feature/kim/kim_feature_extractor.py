import csv
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#infile: a text file containing: line1=user screen name, other lines: 1st element=ot or rt, separated by | then the tweet
#this method generates cosine based features in Kim et al. 2014 (i.e., mean/std of cosine of tweet/retweet pairs)
def feature_tweet_cos(infolder, outcsv):
    with open(outcsv, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["user","ot_cos_mean","ot_cos_std","rt_cos_mean","rt_cos_std"])

        for af in os.listdir(infolder):
            with open(infolder+"/"+af,encoding="utf-8") as f:
                print(af)

                lines = f.read().splitlines()
                user=lines[0]

                ot=[]
                rt=[]
                lines=lines[1:]
                for l in lines:
                    parts=l.split("|",1)
                    if len(parts)<2:
                        print("\tillegal char detected, line ignored:"+l)
                        continue

                    if parts[0]=="ot":
                        ot.append(parts[1])
                    else:
                        rt.append(parts[1])

                if len(ot) == 0:
                    ot_mean=0
                    ot_std=0
                else:
                    ot_cosine = get_cosine_sim(ot)
                    ot_mean=sum(ot_cosine) / len(ot)
                    ot_std=np.std(ot_cosine)
                if len(rt) == 0:
                    rt_mean=0
                    rt_std=0
                else:
                    rt_cosine = get_cosine_sim(rt)
                    rt_mean = sum(rt_cosine) / len(rt)
                    rt_std = np.std(rt_cosine)

                row=[user,ot_mean, ot_std, rt_mean, rt_std]
                csvwriter.writerow(row)

        #calculate pairwise similarities
def get_cosine_sim(strs):
    vectors = [t for t in get_vectors(strs)]
    matrix= cosine_similarity(vectors)
    vec = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            vec.append(matrix[i][j])
    return vec

#create tfidf vectors
def get_vectors(strs):
    vectorizer = CountVectorizer(strs)
    vectorizer.fit(strs)
    return vectorizer.transform(strs).toarray()








if __name__ == "__main__":
    kim_tweets_for_cosine="/home/zz/Work/msm4phi_data/paper2/reported/kim/tweets_for_cosine"
    kim_tweets_for_cosine_features="/home/zz/Work/msm4phi_data/paper2/reported/kim/features_ot_rt_cos.csv"
    feature_tweet_cos(kim_tweets_for_cosine,kim_tweets_for_cosine_features)