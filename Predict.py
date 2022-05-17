import pickle

import joblib
import numpy
import pandas as pd
import nltk
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter
import textstat
from lexicalrichness import LexicalRichness
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

svc_model = joblib.load("./saved_model1.joblib")
dataset = pd.DataFrame(columns=['title', 'text', 'source'], index=range(1))

dataset[
    "text"] = "A serologic test is a blood test that looks for antibodies created by your immune system. There are many reasons you might make antibodies, the most important of which is to help fight infections. The serologic test for COVID-19 specifically looks for antibodies against the COVID-19 virus. Your body takes at least five to 10 days after you have acquired the infection to develop antibodies to this virus. For this reason, serologic tests are not sensitive enough to accurately diagnose an active COVID-19 infection, even in people with symptoms. However, serologic tests can help identify anyone who has recovered from coronavirus. This may include people who were not initially identified as having COVID-19 because they had no symptoms, had mild symptoms, chose not to get tested, had a false-negative test, or could not get tested for any reason. Serologic tests will provide a more accurate picture of how many people have been infected with, and recovered from, coronavirus, as well as the true fatality rate.Serologic tests may also provide information about whether people become immune to coronavirus once they've recovered and, if so, how long that immunity lasts. In time, these tests may be used to determine who can safely go back out into the community.Scientists can also study coronavirus antibodies to learn which parts of the coronavirus the immune system responds to, in turn giving them clues about which part of the virus to target in vaccines they are developing.Serological tests are starting to become available and are being developed by many private companies worldwide. However, the accuracy of these tests needs to be validated before widespread use in the US."
dataset["title"] = "What is serologic (antibody) testing for COVID-19? What can it be used for?"
dataset["source"] = "https://www.health.harvard.edu/"

dataset.text.fillna(dataset.title, inplace=True)
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset.title.fillna('missing', inplace=True)

dataset['title_num_uppercase'] = dataset['title'].str.count(r'[A-Z]')
dataset['text_num_uppercase'] = dataset['text'].str.count(r'[A-Z]')
dataset['text_len'] = dataset['text'].str.len()
dataset['text_pct_uppercase'] = dataset.text_num_uppercase.div(dataset.text_len)

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
dataset['title_num_stop_words'] = dataset['title'].str.split().apply(lambda x: len(set(x) & stop_words))
dataset['text_num_stop_words'] = dataset['text'].str.split().apply(lambda x: len(set(x) & stop_words))
dataset['text_word_count'] = dataset['text'].apply(lambda x: len(str(x).split()))
dataset['text_pct_stop_words'] = dataset['text_num_stop_words'] / dataset['text_word_count']

dataset.drop(['text_num_uppercase', 'text_len', 'text_num_stop_words', 'text_word_count'], axis=1, inplace=True)
dataset['token'] = dataset.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)
dataset['pos_tags'] = dataset.apply(lambda row: nltk.pos_tag(row['token']), axis=1)
tag_count_dataset = pd.DataFrame(dataset['pos_tags'].map(lambda x: Counter(tag[1] for tag in x)).to_list())
dataset = pd.concat([dataset, tag_count_dataset], axis=1).fillna(0).drop(['pos_tags', 'token'], axis=1)

dataset = dataset[['title', 'text', 'source', 'title_num_uppercase', 'text_pct_uppercase', 'title_num_stop_words', 'text_pct_stop_words']].rename(columns={'NNP': 'NNP_title'})

dataset['token'] = dataset.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
dataset['pos_tags'] = dataset.apply(lambda row: nltk.pos_tag(row['token']), axis=1)

print(len(dataset.columns))
tag_count_dataset = pd.DataFrame(dataset['pos_tags'].map(lambda x: Counter(tag[1] for tag in x)).to_list())
print(tag_count_dataset)
original_tags = ['NNS', 'VBP', 'IN', 'DT', 'JJ', 'NN', 'NNP', 'MD', 'VB', 'PRP', 'WRB', 'TO', ',', 'VBZ', 'WDT', 'CC', '.', 'PRP$', 'RB', 'VBG', 'VBD', 'CD', 'WP', 'RBR', 'VBN', 'JJS', 'RP', 'JJR', ':', '``', '(', ')', "''", 'POS', 'EX', 'PDT', 'RBS', 'NNPS', '$', 'WP$', 'FW', 'UH', '#', 'SYM', '.']

df_tags = pd.DataFrame(columns=original_tags, index=range(1))

for tag in original_tags:
    if tag in tag_count_dataset.columns:
        df_tags[tag] = tag_count_dataset[tag]
    else:
        df_tags[tag] = numpy.zeros(1).tolist()

dataset = pd.concat([dataset, df_tags], axis=1).fillna(0).drop(['pos_tags', 'token'], axis=1)

dataset['num_negation'] = dataset['text'].str.lower().str.count(
    "no|not|never|none|nothing|nobody|neither|nowhere|hardly|scarcely|barely|doesn’t|isn’t|wasn’t|shouldn’t|wouldn’t|couldn’t|won’t|can't|don't")
dataset['num_interrogatives_title'] = dataset['title'].str.lower().str.count("what|who|when|where|which|why|how")
dataset['num_interrogatives_text'] = dataset['text'].str.lower().str.count("what|who|when|where|which|why|how")

# TRAINING MODELLO
reading_ease = []
for doc in dataset['text']:
    reading_ease.append(textstat.flesch_reading_ease(doc))

smog = []
for doc in dataset['text']:
    smog.append(textstat.smog_index(doc))

kincaid_grade = []
for doc in dataset['text']:
    kincaid_grade.append(textstat.flesch_kincaid_grade(doc))

liau_index = []
for doc in dataset['text']:
    liau_index.append(textstat.coleman_liau_index(doc))

readability_index = []
for doc in dataset['text']:
    readability_index.append(textstat.automated_readability_index(doc))

readability_score = []
for doc in dataset['text']:
    readability_score.append(textstat.dale_chall_readability_score(doc))

difficult_words = []
for doc in dataset['text']:
    difficult_words.append(textstat.difficult_words(doc))

write_formula = []
for doc in dataset['text']:
    write_formula.append(textstat.linsear_write_formula(doc))

gunning_fog = []
for doc in dataset['text']:
    gunning_fog.append(textstat.gunning_fog(doc))

text_standard = []
for doc in dataset['text']:
    text_standard.append(textstat.text_standard(doc))

dataset['flesch_reading_ease'] = reading_ease
dataset['smog_index'] = smog
dataset['flesch_kincaid_grade'] = kincaid_grade
dataset['automated_readability_index'] = readability_index
dataset['dale_chall_readability_score'] = readability_score
dataset['difficult_words'] = difficult_words
dataset['linsear_write_formula'] = write_formula
dataset['gunning_fog'] = gunning_fog
dataset['text_standard'] = text_standard

print(dataset)

ttr = []
for doc in dataset['text']:
    lex = LexicalRichness(doc)
    ttr.append(lex.ttr)

dataset['ttr'] = ttr
print(len(dataset.columns))

dataset['num_powerWords_text'] = dataset['text'].str.lower().str.count('improve|trust|immediately|discover|profit|learn|know|understand|powerful|best|win|more|bonus|exclusive|extra|you|free|health|guarantee|new|proven|safety|money|now|today|results|protect|help|easy|amazing|latest|extraordinary|how to|worst|ultimate|hot|first|big|anniversary|premiere|basic|complete|save|plus|create')
dataset['num_casualWords_text'] = dataset['text'].str.lower().str.count('make|because|how|why|change|use|since|reason|therefore|result')
dataset['num_tentativeWords_text'] = dataset['text'].str.lower().str.count('may|might|can|could|possibly|probably|it is likely|it is unlikely|it is possible|it is probable|tends to|appears to|suggests that|seems to')
dataset['num_emotionWords_text'] = dataset['text'].str.lower().str.count('ordeal|outrageous|provoke|repulsive|scandal|severe|shameful|shocking|terrible|tragic|unreliable|unstable|wicked|aggravate|agony|appalled|atrocious|corruption|damage|disastrous|disgusted|dreadatasetul|eliminate|harmful|harsh|inconsiderate|enraged|offensive|aggressive|frustrated|controlling|resentful|anger|sad|fear|malicious|infuriated|critical|violent|vindictive|furious|contrary|condemning|sarcastic|poisonous|jealous|retaliating|desperate|alienated|unjustified|violated')


def cleantext(string):
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+", ' ', text)
    text = re.sub(r"www(\S)+", ' ', text)
    text = re.sub(r"&", ' and ', text)
    text = text.replace('&amp', ' ')
    text = re.sub(r"[^0-9a-zA-Z]+", ' ', text)
    text = text.split()
    text = [w for w in text if not w in stop_words]
    text = " ".join(text)
    return text


dataset['text'] = dataset['text'].map(lambda x: cleantext(x))
dataset['title'] = dataset['title'].map(lambda x: cleantext(x))
dataset['source'] = dataset['source'].map(lambda x: cleantext(x))

classes = {"TRUE": 1, "FAKE": 0}


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]


dataset = dataset.drop(['title', 'text', 'source', 'text_standard'], axis=1)
print(dataset)

testset = dataset.to_numpy()

scaler = StandardScaler()
X_test = scaler.fit_transform(testset)

pred = svc_model.predict(X_test)

print("\nPREDIZIONE:")
if pred == 0:
    print("ALERT FAKE NEWS")
else:
    print("NOTIIZA VERA")
