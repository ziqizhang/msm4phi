import numpy
import pandas as pd
from feature import text_feature_extractor as tfe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
    X_new = numpy.concatenate([X, X_2], axis=1)

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
    X_new = numpy.concatenate([X, X_2], axis=1)

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
    X_new = numpy.concatenate([X, X_2], axis=1)

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
    X_new = numpy.concatenate([X, X_2], axis=1)

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
    X_new = numpy.concatenate([X, X_2], axis=1)

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
    X_new = numpy.concatenate([X, X_2], axis=1)

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

    return X_2, y


def create_basic_and_user_url(csv_basic_feature):
    X, y = create_basic(csv_basic_feature)
    url_column = 23

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    df = df.astype(str)
    X_2 = df[:, url_column]

    # change nan values to 0 and urls to 1
    X_2 = numpy.asarray([0 if x == "nan" else 1 for x in X_2])
    X_2 = X_2.reshape((len(X_2),1))
    X_2 = X_2.astype(numpy.float32)

    X_new = numpy.concatenate([X, X_2], axis=1)

    return X_new,y


def create_basic_auto_dict_and_text(csv_basic_feature, folder_other):
    X, y = create_basic_and_autocreated_dictionary(csv_basic_feature,folder_other)

    X_2, _ = create_textfeatures_profile_and_name(csv_basic_feature)

    X_new = numpy.concatenate([X, X_2], axis=1)

    return X_new,y


def create_pca(csv_basic_feature, folder_other, no_dimensions):

    X, y = create_basic_and_autocreated_dictionary(csv_basic_feature,folder_other)
    X_2, _ = create_manual_dict(csv_basic_feature, folder_other)

    X_new = numpy.concatenate([X,X_2], axis=1)

    # standardize the data
    X_new = StandardScaler().fit_transform(X_new)

    #do PCA
    pca = PCA(n_components= no_dimensions)
    X_pca = pca.fit_transform(X_new)

    return X_pca, y


def create_lda_auto_manual_dict_and_basic(csv_basic_feature, folder_other):

    X, y = create_basic_and_autocreated_dictionary(csv_basic_feature,folder_other)
    X_2, _ = create_manual_dict(csv_basic_feature, folder_other)

    X_new = numpy.concatenate([X,X_2], axis=1)

    print(X_new.shape)

    # standardize the data
    X_new = StandardScaler().fit_transform(X_new)

    #do LDA
    lda = LDA()
    X_lda = lda.fit_transform(X_new,y)

    return X_lda, y


def get_all_numeric_features(csv_basic_feature, folder_other):

    # get all numeric features

    X_basic, y = create_basic(csv_basic_feature)

    X_manual, _ = create_manual_dict(csv_basic_feature, folder_other)
    X_auto, _ = create_autocreated_dict(csv_basic_feature, folder_other)

    csv_diseases_in_tweets = folder_other + \
                             "/tweet_feature/diseases_in_tweets.csv"
    df = pd.read_csv(csv_diseases_in_tweets, header=0, delimiter=",", quoting=0).as_matrix()
    X_diseases_tweets = df[:, 2:]
    X_diseases_tweets = X_diseases_tweets.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_diseases_tweets,False)


    csv_topical_tweets = folder_other + \
                         "/tweet_feature/topical_tweets.csv"
    df = pd.read_csv(csv_topical_tweets, header=0, delimiter=",", quoting=0).as_matrix()
    X_topical_tweets = df[:, 2:]
    X_topical_tweets = X_topical_tweets.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_topical_tweets,False)

    csv_manual_g = folder_other + \
                   "/manual_dict_feature_1/features_manual_dict_g.csv"
    df = pd.read_csv(csv_manual_g, header=0, delimiter=",", quoting=0).as_matrix()
    X_manual_g = df[:, 1:]
    X_manual_g = X_manual_g.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_manual_g,False)

    csv_tweet_hashtag = folder_other + \
                        "/dictionary_feature_1/feature_disease_hashtag_match_profile.csv"
    df = pd.read_csv(csv_tweet_hashtag, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_hashtag = df[:, 1:]
    X_tweet_hashtag = X_tweet_hashtag.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_hashtag,False)

    csv_tweet_word = folder_other + \
                     "/dictionary_feature_1/feature_disease_word_match_profile.csv"
    df = pd.read_csv(csv_tweet_word, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_word = df[:, 1:]
    X_tweet_word = X_tweet_word.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_word,False)

    csv_tweet_generic_dict1 = folder_other + \
                              "/dictionary_feature_1/feature_generic_dict_match_name.csv"
    df = pd.read_csv(csv_tweet_generic_dict1, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_generic_dict1 = df[:, 1:]
    X_tweet_generic_dict1 = X_tweet_generic_dict1.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_generic_dict1,False)

    csv_tweet_generic_dict2 = folder_other + \
                              "/dictionary_feature_1/feature_generic_dict_match_profile.csv"
    df = pd.read_csv(csv_tweet_generic_dict2, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_generic_dict2 = df[:, 1:]
    X_tweet_generic_dict2 = X_tweet_generic_dict2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_generic_dict2,False)

    # concatenate all feature sets
    #basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_basic, X_manual, X_auto, X_diseases_tweets, X_topical_tweets, X_manual_g,
                               X_tweet_hashtag, X_tweet_word, X_tweet_generic_dict1, X_tweet_generic_dict2], axis=1)
    print("Size of the array: ")
    print(X_all.shape)

    return X_all, y

def create_pca_all(csv_basic_feature, folder_other, no_dimensions):

    X_all, y = get_all_numeric_features(csv_basic_feature, folder_other)

    # standardize the data
    X_new = StandardScaler().fit_transform(X_all)

    #do PCA
    pca = PCA(n_components= no_dimensions)
    X_pca = pca.fit_transform(X_new)

    return X_pca, y


def create_lda_all(csv_basic_feature, folder_other):

    X_all, y = get_all_numeric_features(csv_basic_feature, folder_other)

    # standardize the data
    X_new = StandardScaler().fit_transform(X_all)

    #do LDA
    lda = LDA()
    X_lda = lda.fit_transform(X_new,y)

    return X_lda, y

def create_pca_and_lda_all(csv_basic_feature, folder_other):

    X_all, y = get_all_numeric_features(csv_basic_feature, folder_other)

    # standardize the data
    X_new = StandardScaler().fit_transform(X_all)

    #do PCA
    pca = PCA(n_components= 50)
    X_pca = pca.fit_transform(X_new)

    #do LDA
    lda = LDA()
    X_lda = lda.fit_transform(X_pca,y)

    return X_lda,y


def create_pca_text_and_autodict(csv_basic_feature, folder_other, no_dimensions):

    X, y = create_basic_auto_dict_and_text(csv_basic_feature, folder_other)

    # standardize the data
    X_new = StandardScaler().fit_transform(X)

    #do PCA
    pca = PCA(n_components= no_dimensions)
    X_pca = pca.fit_transform(X_new)

    return X_pca, y

def create_lda_text_and_autodict(csv_basic_feature, folder_other):

    X, y = create_basic_auto_dict_and_text(csv_basic_feature, folder_other)

    # standardize the data
    X_new = StandardScaler().fit_transform(X)

    #do LDA
    print("before LDA:")
    print(X_new.shape)
    lda = LDA()
    X_lda = lda.fit_transform(X_new,y)

    return X_lda, y

def create_lda_text_and_numeric_all(csv_basic_feature, folder_other):

    X, y = create_textfeatures_profile_and_name(csv_basic_feature)

    X_2, _ = create_lda_all(csv_basic_feature, folder_other)

    X_new = numpy.concatenate([X,X_2], axis=1)

    # standardize the data
    X_new = StandardScaler().fit_transform(X_new)

    #do LDA
    print("before LDA:")
    print(X_new.shape)
    lda = LDA()
    X_lda = lda.fit_transform(X_new,y)

    return X_lda, y




