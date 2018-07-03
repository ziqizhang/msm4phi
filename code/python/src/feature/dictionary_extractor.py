# this code creates dictionary for different types of user profiles automatically
import functools

import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from feature import nlp

text_normalization_option = 1  # 0=stemming, 1=lemma, 2=nothing

text_processor = TextPreProcessor(
    # terms that will be normalized
    # normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
    #                'time', 'url', 'date', 'number'],
    #     # terms that will be annotated
    #     annotate={"hashtag", "allcaps", "elongated", "repeated",
    #               'emphasis', 'censored'},

    normalize=[],
    # terms that will be annotated
    annotate={'elongated',
              'emphasis'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=False).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)



# input is a csv file that must a user at each row, and text to be concatenated to form user profile at different columns
# output: a dictionary containing a bag of profile text (normalised) for each label
def load_user_profile_text(user_features_csv, *col_text):
    df = pd.read_csv(user_features_csv, header=0, delimiter=",", quoting=0
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

            tweet_text = text_processor.pre_process_doc(tweet_text.strip())
            tweet_text = list(filter(lambda a: a != '<elongated>', tweet_text))
            tweet_text = list(filter(lambda a: a != '<emphasis>', tweet_text))
            tweet_text = list(filter(lambda a: a != 'RT', tweet_text))
            tweet_text = list(filter(lambda a: a != '"', tweet_text))
            tweet_text = " ".join(tweet_text)
            if len(tweet_text) > 1:
                proftext.append(tweet_text)
        label_to_proftext[l]=proftext

    return label_to_proftext


def extract_dict(label_to_proftext: dict):
    # frequency based score
    label_vocab_to_totalfreq = dict()
    vocab_overall_frequency = dict()

    label_to_nouns = dict()
    label_to_verbs = dict()

    for label, texts in label_to_proftext.items():
        print(label+","+str(len(texts)))
        vocab_score = dict()

        # identify verbs and nouns for this label
        nouns = set()
        verbs = set()
        for t in texts:
            #count+=1
            #print(count)
            orig_toks = nlp.tokenize(t, 2)
            stem_toks = nlp.tokenize(t, text_normalization_option)
            pos_tags = nlp.get_pos_tags(orig_toks)
            for i in range(0, len(pos_tags)):
                word = orig_toks[i].lower()
                if word in nlp.stopwords or len(word)<2:
                    continue
                stem = stem_toks[i]
                if stem in vocab_score.keys():
                    vocab_score[stem]+=1
                else:
                    vocab_score[stem]=1

                tag = pos_tags[i]
                if tag in ["NN", "NNS", "NNP", "NNPS"]:
                    nouns.add(stem_toks[i])
                elif tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                    verbs.add(stem_toks[i])
            label_to_nouns[label] = nouns
            label_to_verbs[label] = verbs

        label_vocab_to_totalfreq[label] = vocab_score
        for e, frequency in vocab_score.items():
            if frequency==0:
                continue
            if e in vocab_overall_frequency.keys():
                vocab_overall_frequency[e] += frequency
            else:
                vocab_overall_frequency[e] = frequency

    # calculate weighted score
    label_vocab_to_weightedscore = dict()
    for label, vocab_freq in label_vocab_to_totalfreq.items():
        vocab_score = dict()
        for e, frequency in vocab_freq.items():
            if e not in vocab_overall_frequency.keys():
                continue
            totalfreq = vocab_overall_frequency[e]
            s = frequency / totalfreq
            if s==1.0:
                continue
            vocab_score[e] = s
        label_vocab_to_weightedscore[label] = vocab_score

    return label_vocab_to_totalfreq, label_vocab_to_weightedscore, label_to_nouns, label_to_verbs


def create_filtered_dict(outfolder, vocab_with_score: dict,
                         label_to_nouns, label_to_verbs,
                         topN, verb_topN, noun_topN):
    for l, vocab in vocab_with_score.items():
        nouns_of_l = label_to_nouns[l]
        verbs_of_l = label_to_verbs[l]

        nouns = []
        verbs = []
        any = []
        sorted_by_value = sorted(vocab.items(), reverse=True, key=lambda kv: kv[1])

        idx = 0
        while (True and idx<len(sorted_by_value)):
            entry = sorted_by_value[idx]
            value = entry[1]
            entry=entry[0]
            if len(any) < topN:
                any.append(entry + "," + str(value))
            if len(nouns) < noun_topN and entry in nouns_of_l:
                nouns.append(entry + "," + str(value))
            if len(verbs) < verb_topN and entry in verbs_of_l:
                verbs.append(entry + "," + str(value))

            if len(nouns) == noun_topN and len(verbs) == verb_topN and len(any) == topN:
                break

            idx+=1

        # save the created dictionaries for this label
        file = open(outfolder + "/" + l + "_" + str(topN) + "_any.csv", 'w')
        for a in any:
            file.write(a + "\n")
        file = open(outfolder + "/" + l + "_" + str(noun_topN) + "_noun.csv", 'w')
        for n in nouns:
            file.write(n + "\n")
        file = open(outfolder + "/" + l + "_" + str(verb_topN) + "_verb.csv", 'w')
        for v in verbs:
            file.write(v + "\n")


if __name__ == "__main__":
    # col 40-label; col 22-desc;
    profile_text = load_user_profile_text("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/"
                                          "data/stakeholder_classification/annotation/merged_training_data/user_features_and_labels_2.csv",
                                          15) #22)
    vocab_to_totalfreq, vocab_to_weightedscore, label_to_nouns, label_to_verbs = \
        extract_dict(profile_text)

    create_filtered_dict(
        "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/stakeholder_classification/dictionary_feature/frequency",
        vocab_to_totalfreq, label_to_nouns, label_to_verbs,
        200, 100, 100)

    create_filtered_dict(
        "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/stakeholder_classification/dictionary_feature/frequency_rel",
        vocab_to_weightedscore, label_to_nouns, label_to_verbs,
        20, 10, 10)
