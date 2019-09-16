import gensim
import sys
import os
import csv

import numpy
import pandas
from datetime import datetime
from nltk import WordNetLemmatizer
from feature.nlp import stemmer

from gensim import corpora, models

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
    return result

def load_texts(in_folder):
    texts=[]
    for af in os.listdir(in_folder):
        f = open(in_folder+"/"+af, "r")
        lines = f.readlines()
        f.close()
        for i in range(1,len(lines)):
            parts=lines[i].lower().split("|")
            if len(parts)<2:
                continue
            l = parts[1].strip()
            if (l.startswith("rt ")):
                l=l[3:]
            texts.append(l)

def load_texts_simple(in_file):
    f = open(in_file, "r", encoding="utf8", errors="ignore")
    lines = f.readlines()
    return lines

def load_texts_csv(in_file, txt_col, max):
    texts=[]
    count=0
    df = pandas.read_csv(in_file, header=0, delimiter=",", quoting=0, quotechar='"',
                         encoding='utf-8')
    for index, row in df.iterrows():
        text=row[txt_col].replace("\\s+"," ").strip()
        texts.append(text)
        count+=1
        if count>=max:
            break
    return texts


def load_user_profiles(in_folder):
    profiles={}
    for af in os.listdir(in_folder):
        f = open(in_folder + "/" + af, "r")
        lines = f.readlines()
        f.close()

        text = ""
        user = lines[0].strip()
        for i in range(1, len(lines)):
            parts = lines[i].lower().split("|")
            if len(parts) < 2:
                continue
            l = parts[1].strip()
            if (l.startswith("rt ")):
                l = l[3:]
            text+=l+" "
        profiles[user]=text.strip()
    return profiles

def save_as_utf(in_file, out_file):
    f = open(out_file, "w",encoding='utf-8', errors='replace')
    inf=open(in_file, "r",encoding='utf-8', errors='replace')
    while True:
        try:
            x = inf.readline()
        except UnicodeEncodeError:
            continue
        f.write(x)
        if not x: break

    f.close()
    inf.close()


if __name__ == "__main__":
    # save_as_utf(sys.argv[1], sys.argv[1]+".clean")
    # exit(0)

    '''
/home/zz/Work/msm4phi_data/paper2/reported/penn/lda_tweets/sent140.training.1600000.processed.noemoticon.csv
/home/zz/Work/msm4phi_data/paper2/reported/penn/lda_tweets/lda_sent140.model
csv
/home/zz/Work/msm4phi_data/paper2/reported/kim/tweets_for_cosine
/home/zz/Work/msm4phi_data/paper2/reported/penn/lda_sent140_features.csv

/home/zz/Work/msm4phi_data/paper2/reported/penn/lda_tweets/msm4phi_except_train_users
/home/zz/Work/msm4phi_data/paper2/reported/penn/lda_tweets/lda_msm4phi.model
raw
/home/zz/Work/msm4phi_data/paper2/reported/kim/tweets_for_cosine
/home/zz/Work/msm4phi_data/paper2/reported/penn/lda_msm4phi_features.csv
            '''

    # load all the training data
    # documents=load_texts_simple(sys.argv[1])
    # documents = load_texts(sys.argv[1])
    in_data=sys.argv[1]
    lda_out_file=sys.argv[2]
    in_data_format=sys.argv[3]
    in_user_profiles=sys.argv[4]
    out_features=sys.argv[5]

    print("Loading data, at {}".format(datetime.now()))
    if in_data_format == 'csv':
        documents = load_texts_csv(in_data, 5,
                                   10000000)  # /home/zz/Work/msm4phi_data/trainingandtestdata/training.1600000.processed.noemoticon.csv
    else:
        documents = load_texts_simple(in_data)

    print("Extracting vocab, {} texts, at {}".format(len(documents), datetime.now()))
    processed_docs = []
    for d in documents:
        processed_docs.append(preprocess(d))
    print("Buildnig dictionary, {} texts, at {}".format(len(documents), datetime.now()))
    dictionary = gensim.corpora.Dictionary(processed_docs)
    # threshold for filtering dictionary entries
    print("Filtering dictionary, {} texts, at {}".format(len(documents), datetime.now()))
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000000)

    print("Calc BOW, {} texts, at {}".format(len(documents), datetime.now()))
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    print("Calc TFIDF, {} texts, at {}".format(len(documents), datetime.now()))
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    # lda use tfidf, 500 iterations, 100 toics
    print("Running LDA, {} texts, at {}".format(len(documents), datetime.now()))
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=100, id2word=dictionary, iterations=500)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    lda_model_tfidf.save(lda_out_file)

    ##### test
    unseen_document = 'How a Pentagon deal became an identity crisis for Google'
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1]):
         print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, topn=100)))
    #####

    print("Training completed at {}, now applying to test documents... ".format(datetime.now()))
    with open(out_features, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header=["user"]
        for i in range(100):
            header.append(i)
        csvwriter.writerow(header)

        user_profiles=load_user_profiles(in_user_profiles)
        count=0
        for k, v in user_profiles.items():
            bow_vector = dictionary.doc2bow(preprocess(v))
            topics=lda_model_tfidf[bow_vector]
            weights=numpy.zeros(100)

            for entry in topics:
                weights[entry[0]]=entry[1]
            row=[k]
            row.extend(list(weights))
            csvwriter.writerow(row)
            count+=1
            print(count)

    # unseen_document = 'How a Pentagon deal became an identity crisis for Google'
    # bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    # for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1]):
    #     print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, topn=100)))
    print("complete")