import csv
import logging

import sys

import pandas
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from feature import nlp
from datetime import datetime
import numpy
import random as rn
import gensim
import os
import pickle


logger = logging.getLogger(__name__)

word_vectorizer = CountVectorizer(
            # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            tokenizer=nlp.tokenize,
            preprocessor=nlp.normalize_tweet,
            ngram_range=(1, 1),
            stop_words=nlp.stopwords,  # We do better when we keep stopwords
            decode_error='replace',
            min_df=5,
            max_features=40000
        )

def load_embedding_model(embedding_model_file):
    gensimFormat = ".gensim" in embedding_model_file
    if gensimFormat:
        pretrained_embedding_models = gensim.models.KeyedVectors.load(embedding_model_file, mmap='r')
    else:
        pretrained_embedding_models = gensim.models.KeyedVectors. \
            load_word2vec_format(embedding_model_file, binary=True)
    return pretrained_embedding_models

def get_words(texts):
    logger.info("\tprocesses texts to extract words, {}".format(datetime.now()))
    word_count = word_vectorizer.fit_transform(texts)
    logger.info("\t\t complete, dim={}, {}".format(word_count.shape, datetime.now()))
    vocab = {v: i for i, v in enumerate(word_vectorizer.get_feature_names())}
    index = {v: k for k, v in vocab.items()}

    return vocab, index

def populate_word_vectors(word_emb_model, word_vocab, expected_emb_dim):
    randomized_vectors = {}
    vector_index=[]
    word_vectors = numpy.zeros((len(word_vocab), expected_emb_dim))
    count = 0
    random = 0
    for word, i in word_vocab.items():
        vector_index.append(i)
        if word in word_emb_model.wv.vocab.keys():
            vec = word_emb_model.wv[word]
            word_vectors[i] = vec
        else:
            random += 1
            if word in randomized_vectors.keys():
                vec = randomized_vectors[word]
            else:
                max = len(word_emb_model.wv.vocab.keys()) - 1
                index = rn.randint(0, max)
                word = word_emb_model.index2word[index]
                vec = word_emb_model.wv[word]
                randomized_vectors[word] = vec
            word_vectors[i] = vec
        count += 1
        if count % 100000 == 0:
            print(count)

    return word_vectors, vector_index

def calc(word_vectors):
    matrix = cosine_similarity(word_vectors)
    return matrix

def cluster(similarity_matrix):
    #cls=SpectralClustering(n_clusters=200).fit_predict(similarity_matrix)
    cls = KMeans(n_clusters=200).fit_predict(similarity_matrix)

    #create word cluster memberships
    cluster_member={}
    for i in range(len(cls)):
        cluster=cls[i]
        if cluster in cluster_member.keys():
            members=cluster_member[cluster]
            members.append(i)
        else:
            members=[i]
            cluster_member[cluster]=members

    return cluster_member

def generate_cluster_features(input_folder,cluster_members:dict,
                              word_vocab:dict, user_tweets_folder,
                              raw_label_data,
                              out_feature_csv):

    df = pandas.read_csv(raw_label_data, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()

    user_tweets={}
    for af in os.listdir(user_tweets_folder):
        f = open(input_folder + "/" + af, "r")
        lines = f.readlines()
        f.close()

        user = lines[0].strip()
        tweets = ""
        for i in range(1, len(lines)):
            parts = lines[i].lower().split("|")
            if len(parts) < 2:
                continue
            l = parts[1].strip()
            if (l.startswith("rt ")):
                l = l[3:]
            tweets += l + " "
        tweets = tweets.strip()
        user_tweets[user]=tweets

    with open(out_feature_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header=["user"]
        for i in range(200):
            header.append(i)
        header.append("label")
        csvwriter.writerow(header)

        for r in df:
            user = r[14]
            label = r[40]

            row=[]
            features=numpy.zeros(200)
            row.append(user)

            if user in user_tweets.keys():
                tweets = user_tweets[user]
            else:
                tweets=r[22] #if this user has no tweets, use its profile
                if type(tweets) is float:
                    tweets=""

            tweets=nlp.normalize_tweet(tweets)
            toks=nlp.tokenize(tweets)
            for t in toks:
                if t in nlp.stopwords:
                    continue

                if t in word_vocab.keys():
                    index=word_vocab[t]
                    cluster=cluster_members[index]
                    features[cluster]=features[cluster]+1


            row.extend(list(features))
            row.append(label)
            csvwriter.writerow(row)

if __name__ == "__main__":
    #1. read all tweets
    #glove.840B.300d.bin.gensim
    #GoogleNews-vectors-negative300.bin.gz
    # f = open('clusters.data', 'rb')
    # cluster_members = pickle.load(f)
    # f.close()

    input_folder=sys.argv[1]
    tweets=set()

    #loading tweets from msm4phi corpus
    # for af in os.listdir(input_folder):
    #     f = open(input_folder+"/"+af, "r")
    #     lines = f.readlines()
    #     f.close()
    #     for i in range(1,len(lines)):
    #         parts=lines[i].lower().split("|")
    #         if len(parts)<2:
    #             continue
    #         l = parts[1].strip()
    #         if (l.startswith("rt ")):
    #             l=l[3:]
    #         tweets.add(l)

    #loading tweets from sent140
    df = pandas.read_csv(input_folder, header=0, delimiter=",", quoting=0, quotechar='"',
                         encoding='utf-8')
    for index, row in df.iterrows():
        text = row[5].replace("\\s+", " ").strip()
        tweets.add(text)


    #2. extract vocab
    print("Extracting vocab, {} texts, at {}".format(len(tweets), datetime.now()))
    word_vocab,word_index=get_words(list(tweets))
    #
    #3. load embedding model
    print("Loading embeddings, at {}".format(datetime.now()))
    word_emb=load_embedding_model(sys.argv[2])

    #4. populate word vectors
    print("Populating word vectors, {} words, at {}".format(len(word_vocab), datetime.now()))
    word_vectors, vector_index=populate_word_vectors(word_emb, word_vocab, 300)

    #5. similarity matrix
    print("Calculating simmatrix, at {}".format(datetime.now()))
    similarity=calc(word_vectors)

    #6. clustering
    print("Clustering, at {}".format(datetime.now()))
    clusters = cluster(similarity)
    cluster_members={}
    for k,v in clusters.items():
        for index in v:
            cluster_members[index]=k

    filehandler = open("clusters.data", 'wb')
    pickle.dump(cluster_members, filehandler)
    filehandler.close()



    #7. create features
    print("Creating feature csv, at {}".format(datetime.now()))
    generate_cluster_features(input_folder,cluster_members,word_vocab,input_folder,
                              sys.argv[3], sys.argv[4])

    print("complete")