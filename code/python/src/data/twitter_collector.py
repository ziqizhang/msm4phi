import logging
import random
import sys
import json
import os
import traceback
import urllib.request
import pandas as pd
import time
import urllib.request

import pickle
import datetime

import tweepy
from SolrClient import SolrClient
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener, Stream


IGNORE_RETWEETS = False
LANGUAGES_ACCETED = ["en"]
SOLR_CORE_SEARCHAPI = "msm4phi"
TWITTER_TIME_PATTERN = "%a %b %d %H:%M:%S %z %Y"
SOLR_TIME_PATTERN = "%Y-%m-%dT%H:%M:%SZ"  # YYYY-MM-DDThh:mm:ssZ
LOG_DIR = os.getcwd() + "/logs"
logger = logging.getLogger(__name__)
logging.basicConfig(filename=LOG_DIR + '/twitter_stream.log', level=logging.INFO, filemode='w')
# feat_vectorizer=fv_chase_basic.FeatureVectorizerChaseBasic()
SCALING_STRATEGY = -1


SOLR_CORE_TWEETS= "tweets"


def commit(core_name, solr_url):
    code = urllib.request. \
        urlopen("{}/{}/update?commit=true".
                format(solr_url, core_name)).read()

def read_auth(file):
    vars = {}
    with open(file) as auth_file:
        for line in auth_file:
            name, var = line.partition("=")[::2]
            vars[name.strip()] = str(var).strip()
    return vars


def read_search_criteria(file):
    vars = {}
    with open(file) as auth_file:
        for line in auth_file:
            name, var = line.partition("=")[::2]
            vars[name.strip()] = str(var).strip()
    return vars


class TwitterStream(StreamListener):
    #https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
    __solr = None
    __core = None
    __count = 0
    __count_retweet = 0

    def __init__(self, solr_url):
        super().__init__()
        self.__solr = SolrClient(solr_url)
        self.__core = SOLR_CORE_TWEETS

    def on_data(self, data):
        self.__count += 1
        if self.__count % 200 == 0:
            code = commit(SOLR_CORE_TWEETS, self.__solr)
            now = datetime.datetime.now()
            print("{} processed: {}".
                  format(now, self.__count))
            logger.info("{} processed: {}".
                        format(now, self.__count))
        jdata = None
        try:
            jdata = json.loads(data)

            if jdata is not None and "id" in jdata.keys():
                # created_at_time
                str_created_at = jdata["created_at"]
                time = datetime.datetime.strptime(str_created_at, TWITTER_TIME_PATTERN)
                str_solr_time = time.utcnow().strftime(SOLR_TIME_PATTERN)

                doc={'id': jdata["id_str"],
                         'created_at': str_solr_time,
                         'lang': jdata["lang"],
                         'status_text': jdata["extended_tweet"]["full_text"]
                         }
                self.collect_tweet_entities(doc,jdata)
                self.collect_tweet_quote_info(doc,jdata)
                self.collect_retweet_info(doc,jdata)
                self.collect_tweet_favorite_info(doc,jdata)
                self.collect_tweet_location_info(doc,jdata)
                self.collect_tweet_reply_info(doc,jdata)
                self.collect_tweet_user_info(doc,jdata)

                self.__solr.index(self.__core, [doc])
        except Exception as exc:
            traceback.print_exc(file=sys.stdout)
            print("Error encountered for {}, error:{} (see log file for details)".format(self.__count, exc))
            if jdata is not None and "id" in jdata.keys():
                tweet_id = jdata["id"]
            else:
                tweet_id = "[failed to parse]"
            logger.info("Error encountered for counter={}, tweet={}, error:{} (see log file for details)".
                        format(self.__count, tweet_id, exc))
            if jdata is not None:
                file = LOG_DIR + "/" + str(tweet_id) + ".txt"
                logger.info("\t input data json written to {}".format(file))
                with open(file, 'w') as outfile:
                    json.dump(jdata, outfile)
            pass
        return (True)

    def on_error(self, status):
        print(status)

    def on_status(self, status):
        print(status.text)

    def collect_tweet_entities(self, doc:dict, tweet_json:dict):
        ##################### tweet entities ###################
        # entities hashtags
        entities=tweet_json["extended_tweet"]["entities"]
        hashtags = entities["hashtags"]
        hashtag_list = []
        for hashtag in hashtags:
            hashtag_list.append(hashtag["text"].lower())

        # entities urls
        urls = entities["urls"]
        url_list = []
        for url in urls:
            url_list.append(url["expanded_url"])

        # entities symbols
        symbols = entities["symbols"]
        symbols_list = []
        for symbol in symbols:
            symbols_list.append(symbol["text"])

        # entities user_mentions
        user_mentions = entities["user_mentions"]
        user_mention_list = []
        for um in user_mentions:
            user_mention_list.append(um["id"])

        # media todo
        ext_entities = tweet_json["extended_tweet"]["extended_entities"]
        if ext_entities is not None:
            media=ext_entities["media"]
            if media is not None:
                doc['media_url']=media["media_url"]
                doc['media_type']=media["media_type"]

        doc['entities_hashtag']= hashtag_list
        doc['entities_symbol']= symbols_list
        doc['entities_url']= url_list
        doc['entities_user_mention']= user_mention_list

    def collect_tweet_quote_info(self,doc:dict, tweet_json:dict):
        #################  quote ####################
        # quoted status id if exists
        if "quoted_status_id_str" in tweet_json:
            quoted_status_id = tweet_json["quoted_status_id_str"]
        else:
            quoted_status_id = None
        doc['quoted_status_id_str']= quoted_status_id
        doc['is_quote_status']=tweet_json["is_quote_status"]
        doc['quote_count']=tweet_json["quote_count"] #nullable


    def collect_tweet_reply_info(self,doc:dict, tweet_json:dict):
        doc['in_reply_to_screen_name']= tweet_json["in_reply_to_screen_name"] #nullable
        doc['in_reply_to_status_id_str']= tweet_json["in_reply_to_status_id_str"]#nullable
        doc['in_reply_to_user_id_str']= tweet_json["in_reply_to_user_id_str"]#nullable
        doc['reply_count']=tweet_json["reply_count"]

    def collect_retweet_info(self,doc:dict, tweet_json:dict):
        doc['retweet_count']= tweet_json["retweet_count"]
        doc['retweeted']= tweet_json["retweeted"]
        doc['retweeted_status_id_str'] =tweet_json["retweeted_status"]["id_str"] #nullable

    def collect_tweet_favorite_info(self,doc:dict, tweet_json:dict):
        doc['favorite_count']=tweet_json["favorite_count"] #nullable

    def collect_tweet_user_info(self,doc:dict, tweet_json:dict):
        doc['user_id_str']= tweet_json["user"]["id_str"]
        doc['user_screen_name']= tweet_json["user"]["screen_name"]
        doc['user_statuses_count']= tweet_json["user"]["statuses_count"]
        doc['user_friends_count']=tweet_json["user"]["friends_count"]
        doc['user_followers_count']= tweet_json["user"]["followers_count"]

    def collect_tweet_location_info(self,doc:dict, tweet_json:dict):
        # place exists
        place = tweet_json["place"]
        if place is not None:
            place_full_name = place["full_name"]
            place_coordinates = place['bounding_box']['coordinates'][0][0]
        else:
            place_full_name = None
            place_coordinates = None

        coordinates = tweet_json["coordinates"]
        # user_location, only compute geocode if other means have failed

        if coordinates == None:
            coordinates = place_coordinates

        coord_lat = None
        coord_lon = None
        if coordinates is not None and len(coordinates) > 0:
            coord_lat = coordinates[0]
            coord_lon = coordinates[1]

        doc['coordinate_lat']=coord_lat
        doc['coordinate_lon']=coord_lon
        doc['place_full_name']=place_full_name
        doc['place_coordinates']=place_coordinates

def index_data(in_file, tweepy_api, tweet_id_col, tweet_label_col):
    start=0

    data=pd.read_csv(in_file, sep=',', encoding="utf-8")
    index=0
    missed=0
    for row in data.itertuples():
        if index<start:
            index+=1
            continue
        tweetid=str(row[tweet_id_col])
        label=row[tweet_label_col]
        if label!='2':
            label='0'

        try:
            tweet = tweepy_api.get_status(tweetid)
            text=tweet.text

            index+=1
            if index%100==0:
                print(index)
        except tweepy.error.TweepError:
            traceback.print_exc(file=sys.stdout)
            missed+=1
            print("==="+str(tweetid)+","+label)
        time.sleep(1)


if __name__=="__main__":
    oauth = read_auth(sys.argv[1])
    googleauth=read_auth(sys.argv[3])
    print(sys.argv[1])
    sc = read_search_criteria(sys.argv[2])
    print(sys.argv[2])
    auth = OAuthHandler(oauth["C_KEY"], oauth["C_SECRET"])
    auth.set_access_token(oauth["A_TOKEN"], oauth["A_SECRET"])


    api=tweepy.API(auth)
    # ===== streaming =====
    twitterStream = Stream(auth, TwitterStream(sys.argv[4]))
    twitterStream.filter(track=[sc["KEYWORDS"]], languages=LANGUAGES_ACCETED)

    # ===== index existing data =====
    # index_data("/home/zqz/Work/chase/data/ml/public/w+ws/labeled_data.csv",
    #            api, 1,2)


    # searcher = TwitterSearch(auth)
    # searcher.index(["#refugeesnotwelcome","#DeportallMuslims", "#banislam","#banmuslims", "#destroyislam",
    #                 "#norefugees","#nomuslims"])
