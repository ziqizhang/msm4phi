import numpy
import pandas as pd
from feature import text_feature_extractor as tfe


# this is a text based feature loader
def create_textfeatures_profile(csv_basic_feature):
    text_col = 22  # 16-names; 22-profiles
    label_col = 40

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    y = df[:, label_col]

    df.astype(str)

    texts = df[:, text_col]
    texts = ["" if type(x) is float else x for x in texts]
    # Convert feature vectors to float64 type
    X_ngram, vocab = tfe.get_ngram_tfidf(texts)

    return X_ngram, y


def create_textfeatures_profile_and_name(csv_basic_feature):
    profile_col = 22  # 16-names; 22-profiles
    name_col = 15
    label_col = 40

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    y = df[:, label_col]

    df.astype(str)

    profiles = df[:, profile_col]
    profiles = ["" if type(x) is float else x for x in profiles]

    names = df[:, name_col]
    names = ["" if type(x) is float else x for x in names]
    names = [str(i) for i in names]
    # Convert feature vectors to float64 type
    profile_ngram, vocab = tfe.get_ngram_tfidf(profiles)
    name_ngram, vocab = tfe.get_ngram_tfidf(names)

    X_ngram = numpy.concatenate([name_ngram, profile_ngram], axis=1)

    return X_ngram, y


def create_textfeatures_name(csv_basic_feature):
    text_col = 16  # 16-names; 23-profiles
    label_col = 40

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    y = df[:, label_col]

    texts = df[:, text_col]
    # Convert feature vectors to float64 type
    X_ngram, vocab = tfe.get_ngram_tfidf(texts)

    return X_ngram, y


# this is the basic feature loader, using only the stats from indexed data.
def create_basic(csv_basic_feature):
    feature_start_col = 1
    feature_end_col = 13
    label_col = 40

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    y = df[:, label_col]

    X = df[:, feature_start_col:feature_end_col + 1]
    # Convert feature vectors to float64 type
    X = X.astype(numpy.float32)

    return X, y


def create_basic_and_diseases_in_tweets(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_diseases_in_tweets = folder_other + \
                                   "/tweet_feature/diseases_in_tweets.csv"

    df = pd.read_csv(csv_diseases_in_tweets, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 2:]
    X_2 = X_2.astype(numpy.float32)
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating

    return X_new, y

def create_basic_and_topical_tweets(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_topical_tweets = folder_other + \
                                   "/tweet_feature/topical_tweets.csv"
    # csv_disease_hashtag_match_profile = folder_other + \
    #                                     "/dictionary_feature_1/feature_disease_hashtag_match_profile.csv"
    # csv_disease_word_match_profile = folder_other + \
    #                                  "/dictionary_feature_1/feature_disease_word_match_profile.csv"
    # csv_generic_dict_match_name = folder_other + \
    #                               "/dictionary_feature_1/feature_generic_dict_match_name.csv"
    # csv_generic_dict_match_profile = folder_other + \
    #                                  "/dictionary_feature_1/feature_generic_dict_match_profile.csv"
    df = pd.read_csv(csv_topical_tweets, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 2:]
    X_2 = X_2.astype(numpy.float32)
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating

    # Add disease_hashtag_match
    # df2 = pd.read_csv(csv_generic_dict_match_profile , header=0, delimiter=",", quoting=0).as_matrix()
    # X_3 = df2[:, 1:]
    # X_3 = X_3.astype(numpy.float32)
    # numpy.nan_to_num(X_3,False)
    # X_new = numpy.concatenate([X_new, X_3], axis=1)

    #Add

    return X_new, y


# def create_basic_diseases_and_topical_tweets(csv_basic_feature, folder_other):
#     X, y = create_basic_and_autocreated_dictionary(csv_basic_feature, folder_other)
#     csv_autocreated_dict_feature = folder_other + \
#                                    "/tweet_feature/topical_tweets.csv"
#     df = pd.read_csv(csv_autocreated_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
#     X_2 = df[:, 2:]
#     X_2 = X_2.astype(numpy.float32)
#     X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
#     # remember to call 'astype' like above on them before concatenating
#
#     return X_new, y


def create_basic_and_manual_dictionary(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_manual_dict_feature = folder_other + \
                              "/manual_dict_feature_1/feature_manualcreated_dict_match_profile.csv"
    df = pd.read_csv(csv_manual_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating
    return X_new, y

def create_basic_and_manual_dictionary_g(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_manual_dict_feature = folder_other + \
                              "/manual_dict_feature_1/features_manual_dict_g.csv"
    df = pd.read_csv(csv_manual_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)

    return X_2, y

def create_basic_and_autocreated_dictionary(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_auto_dict_feature = folder_other + \
                              "/dictionary_feature_1/feature_autocreated_dict_match_profile.csv"
    df = pd.read_csv(csv_auto_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating
    return X_new, y

def create_basic_and_hashtag_match_profile(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_auto_dict_feature = folder_other + \
                            "/dictionary_feature_1/feature_disease_hashtag_match_profile.csv"
    df = pd.read_csv(csv_auto_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating
    return X_new, y

def create_basic_and_word_match_profile(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_auto_dict_feature = folder_other + \
                            "/dictionary_feature_1/feature_disease_word_match_profile.csv"
    df = pd.read_csv(csv_auto_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating
    return X_new, y

def create_basic_and_generic_dict_match_name(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_auto_dict_feature = folder_other + \
                            "/dictionary_feature_1/feature_generic_dict_match_name.csv"
    df = pd.read_csv(csv_auto_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating
    return X_new, y

def create_basic_and_generic_dict_match_profile(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_auto_dict_feature = folder_other + \
                            "/dictionary_feature_1/feature_generic_dict_match_profile.csv"
    df = pd.read_csv(csv_auto_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating
    return X_new, y




def create_manual_dict(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)

    csv_manual_dict_feature = folder_other + \
                              "/manual_dict_feature_1/feature_manualcreated_dict_match_profile.csv"
    df = pd.read_csv(csv_manual_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)
    #X_new = numpy.concatenate([X_2, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating

    return X_2, y

def create_autocreated_dict(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)

    csv_autocreated_dict_feature = folder_other + \
                                   "/dictionary_feature_1/feature_autocreated_dict_match_profile.csv"
    df = pd.read_csv(csv_autocreated_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)
    #X_new = numpy.concatenate([X_2, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating

    return X_2, y


