import numpy as np
import pandas as pd
import scipy
from nltk.tokenize import word_tokenize
import gensim
from flaskexample.preprocess import nlp_preprocess_text, find_features
from sklearn.metrics.pairwise import cosine_similarity
import xgboost
from sklearn.preprocessing import LabelEncoder
import fasttext


def generate_responses(input_question, fasttext_model, train_df, xgboost_model, count_cutoff):
    """
    Generate Top5 responses (or less)
    
    :type string: input_question
    :type fasttext model: fasttext_model
    :type pandas DataFrame: train_df
    :type xgboost model: xgboost_model
    :rtype : filtered_msg
    :rtype list: filtered_reply_text_list
    :rtype list: filtered_reply_cosine_list
    :rtype int: count
    """ 
    # preprocess input response   
    input_question_proc = nlp_preprocess_text(input_question)

    # generate fasttext vector
    input_ff = get_vec(fasttext_model, input_question_proc)

    # generate top20 responses's index and cosine value
    top_indices, top_cosine = compare_cos(train_df, input_ff)
    filtered_reply_text_list, filtered_reply_cosine_list, count = filter_reply_msg(top_indices, top_cosine, 0.16, count_cutoff)
    # preprocess input response to use in xgboost and get sentiment
    y_input, input_sent = find_features(input_question_proc, input_ff)

    # generate a response from xgboost
    xgboost_reply = xgboost_category(xgboost_model, y_input)
    count += 1

    # generate a response from sentiment analysis
    if input_sent < -.3:
        sent_reply = ["Don't be discouraged! Take a look at some of the sample strategies to consider, they might help you succeed next time.",
        "It sounds like you are having some difficulties. Take a look at the sample strategies, or come to see me for some extra help."]
        count += 2
    elif input_sent > .8:
        sent_reply = ["I'm glad to hear that you feel really positively about your work! What strategies did you use that helped you succeed?",
        "I'm happy to hear that you are feeling positive about your progress this week. What worked for you this week? Can you think of other strategies that can continue to help you improve?"]
        count +=2
    else:
        sent_reply = ["What actions have you taken, or can you take to improve? Try looking at some of the sample strategies for ideas of what to try.",
        "Can you think about the strategies you used or will use, and how they can help you improve?"]
        count +=2

    if len(filtered_reply_text_list) == 0:
        count = 0 

    if(input_question == "" or input_question == " "):
        count = 0

    print(top_cosine)
    return filtered_reply_text_list, xgboost_reply, sent_reply, count




# simple function that returns the fasttext vectors for an input
def get_vec(model,text):
    fasttext_model = model
    text_vec  = fasttext_model[text]
    return text_vec

def compare_cos(df, text):
    '''
    compares the vector of the input to all of the vectors in the training data, computes the cosine
    difference between the vectors and selects the top 3 unique responses
    :rtype numpy.ndarray: top_indices
    :rtype list: top_cosine 
    '''
    a = np.array([scipy.spatial.distance.cosine(u, text) for u in df['fasttext']])
    top_n = 1   
    b= a.argsort()[:top_n]
    top_indices = df.iloc[b].feedback.values[0]
    top_cosine = a[b]
    return top_indices, top_cosine


def filter_reply_msg(top_indices, top_cosine, cosine_cut_off, count_cutoff):
    """
    Filter out responses
    1. cosine similarity < cosine_cut_off
    2. too short replies (word count <= count_cutoff)
    3. replies contain other special characters
    
    :type pandas DataFrame: response_df
    :type float: cosine_cut_off
    :type int: count_cutoff
    :rtype list: filtered_msg
    :rtype list: filtered_msg_cosine
    :rtype int: count
    """   
    filtered_msg = []
    filtered_msg_cosine = []
    count = 0
    if top_cosine > cosine_cut_off:
        count = 0
    else:
        filtered_msg = top_indices
        filtered_msg_cosine.append(top_cosine)
        count += 1
    return filtered_msg, filtered_msg_cosine, count

def xgboost_category(model, df):
    df=df.values
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('./flaskexample/file/classes.npy')
    y = model.predict(df)
    y_val = label_encoder.inverse_transform(y)
    return y_val[0]
