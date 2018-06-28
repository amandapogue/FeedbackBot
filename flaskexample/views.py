from flaskexample.fasttext_model import generate_responses
from flask import request
from flask import render_template
from flask import json
from flask import jsonify
from flaskexample import app
from sklearn.externals import joblib
import pandas as pd
import json
import pickle
import fasttext


# Loading the saved fasttext model from disk
fasttext_model = fasttext.load_model('./flaskexample/file/fasttext_model.bin')

# Loading the saved train data frame
train_df_pkl = open('./flaskexample/file/feedback_tbl.pkl', 'rb')
train_df = pickle.load(train_df_pkl)

# Loading the saved xgboost model
xgboost_model = joblib.load('./flaskexample/file/xgboostOver.pkl')


@app.route('/')


@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/generate_response',  methods=['POST'])
def generate_response():
    customer_question = request.form['customer_question']
    # generate response by predefined QA
    count_qa = 0
    
    # generate response based on corpus search (ir: information retrival)
    reply_ft_list, reply_xb_list, reply_sent_list, count_ir = generate_responses(customer_question, fasttext_model, train_df, xgboost_model, 2-count_qa)
    
    count = count_qa + count_ir

    # generate result
    if customer_question == "hi" or customer_question == "hello" or customer_question == "hey" or customer_question == "yo":
        agent_replies = [{'reply_text' : "Hello! Did you forget to put in your reflection? Please take a look at the sample strategies, or feel free to stop by and chat with me about your reflections."},
        {'reply_text':'Hi! Don’t forget to complete your reflections! Try looking at some of our sample strategies for ideas on what to try.'},
        {'reply_text' : "Reflections help you keep track of the strategies you’ve tried. Try looking at some of our sample strategies for ideas on what to try."}]
    else:
        agent_replies = [{'reply_text' : reply_ft_list }, {'reply_text': reply_xb_list}, 
        {'reply_text': reply_sent_list[0]},{'reply_text': reply_sent_list[1]}]
    
    if customer_question == "" or customer_question == " ":
        sent_replies = [{'sent_text' : "Did you forget to put in your reflection? Please take a look at the sample strategies, or feel free to stop by and chat with me about your reflections."},
        {'sent_text':'Don’t forget to complete your reflections! Try looking at some of our sample strategies for ideas on what to try.'},
        {'sent_text' : "Reflections help you keep track of the strategies you’ve tried. Try looking at some of our sample strategies for ideas on what to try."}]
    else:
        sent_replies = [{'sent_text' :'Sorry, I\'m not sure I understand your reflection. Please take a look at the sample strategies, or feel free to stop by and chat with me about your reflections.'},
        {'sent_text': 'Hi. Please take a look at the sample strategies, or feel free to stop by and chat with me about your reflections.'},
        {'sent_text': reply_sent_list[0]},{'sent_text': reply_sent_list[1]}]

    ret = {
        'customer_question': customer_question,
        'agent_replies': agent_replies,
        'sent_replies': sent_replies,
        'count': count
    }

    return jsonify(ret)

    