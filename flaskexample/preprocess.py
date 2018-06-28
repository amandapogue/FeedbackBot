import numpy as np
import pandas as pd

# NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk import wordnet
from nltk.corpus import stopwords
from autocorrect import spell


# a function that removes a bunch of expected things that might be included
# also lower-cases the texts and removes extra spaces
def standardize_new_text(text_field):
    text_field = text_field.replace(r"http\S+", "")
    text_field = text_field.replace(r"http", "")
    text_field = text_field.replace(r"@\S+", "")
    text_field = text_field.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    text_field = text_field.replace(r"@", "at")
    text_field = text_field.replace(r"?", "")
    text_field = text_field.lower()
    text_field = text_field.strip()
    return text_field

# calls the standardizer before it tokenizes the text, then also corrects for simple spelling
# errors. Does not remove stop words, and uses a lemmatizer instead of a stemmer
def nlp_preprocess_text(text_proc):
    lemmatizer = wordnet.WordNetLemmatizer()
    text_proc = standardize_new_text(text_proc)
    cleaned_tokens = []
    tokens = word_tokenize(text_proc.lower())
    for token in tokens:
      #  if token not in stop_words:
        if len(token) > 0 and len(token) < 20: # removes non words
            if not token[0].isdigit() and not token[-1].isdigit(): # removes numbers
                token = spell(token)
                lemmed_tokens = lemmatizer.lemmatize(token)
                cleaned_tokens.append(lemmed_tokens)
                

    text_nlp_proc = ' '.join(wd for wd in cleaned_tokens)

    return text_nlp_proc

# preprocesses the input text to be in the format to be used in the 
# xgboost model
def find_features(text, ff):
    df= pd.DataFrame()
    df = df.append({'text': text,'fasttext' : ff,'tokens' : text}, ignore_index=True)
    df['tokens'] = df.apply(lambda row: word_tokenize(row['text']), axis=1)
    ft = df['fasttext'].iloc[0]
    for x in range(100):
        col_name = 'f_{}'.format(x)
        df[col_name]= ft[x]
    df = build_features(df)
    input_sent = sentiment_text(text)
    df['sentiment'] = input_sent
    df = df.drop(columns = ['text','tokens', 'fasttext', 'pos'])
    return df, input_sent

# feature keywords
pacing = ["rush", "finish", "steps", "time", "done"]
specific = ["try", "tried", "same", "keep", "more", "write", "writing", "work", 
            "study", "organizer","understand","strategy","stragies", "rade", "summary",
           "summarize", "highlight", "already", "listen", "actually", "learn", "learned"]
practice = ["practice", "practicing", "practiced"]
study = ["study", "studied", "read", "reading"]
distractions = ["distraction", "distractions", "attention", "listen", "focus"]
try_ = ["try", "trying", "tried", "work", "worked"]
lack_of_know = ["don", "no", "lack", "lacking"]

# finding the intersection between two lists
def Intersection(lst1, lst2):
    x = len(set(lst1).intersection(lst2))
    if x == 0:
        return 0
    else:
        return 1

def build_features(df):
    df['length'] = df['tokens'].apply(len)
    df['pacing'] = df['tokens'].apply(lambda row: Intersection(pacing,row))
    df['specific'] = df['tokens'].apply(lambda row: Intersection(specific,row))
    df['practice'] = df['tokens'].apply(lambda row: Intersection(practice,row))
    df['study'] = df['tokens'].apply(lambda row: Intersection(study,row))
    df['lack_of_know'] = df['tokens'].apply(lambda row: Intersection(lack_of_know,row))
    df['distractions'] = df['tokens'].apply(lambda row: Intersection(distractions,row))
    df['try'] = df['tokens'].apply(lambda row: Intersection(try_,row))
    df['pos'] = df.apply(lambda row: nltk.pos_tag(row['tokens']), axis=1)
    df['pt_vb'] = df['pos'].apply(lambda x: sum([1 for i in x if i[1] == 'VBD']))
    return df

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='./flaskexample/file/stuff.json'

# Detects and returns the sentiment score for the text
def sentiment_text(text):
    from google.cloud import language 
    import six
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    # Reads the text as plain text
    document = language.types.Document(
        content=text,
        language='en',
        type='PLAIN_TEXT')

    # Detects sentiment in the document. 
    sentiment = client.analyze_sentiment(document).document_sentiment
    return sentiment.score