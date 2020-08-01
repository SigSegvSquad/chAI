from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import nltk
import random
import json
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
words= pickle.load(open('words.pkl', 'rb'))
classes= pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open("intents.json",encoding="utf8"))

app = Flask(__name__)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, context, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    bag.append(int(context))
    return(np.array(bag))

def best_pred(sentence, context):
    p = bow(sentence, words, context, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    tag = return_list[0]['intent']
    probability=return_list[0]['probability']
    return tag, probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/',methods=['GET'])
def predict():
    sentence = request.args.get("msg")
    context = request.args.get("context")

    tag,probability= best_pred(sentence,context)
    tag_base, probability_base= best_pred(sentence, "0")
    print(f'contextual=> context:{tag} probability: {probability}')
    print(f'0_contextual=> context:{tag_base} probability: {probability_base}')

    if probability>= probability_base:
        pass
    else:
        tag=tag_base
        probability=probability_base

    list_of_intents = intents['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            context=i["tag"]
            break
    response={
        "response":result,
        "context":context,
        "probability":probability
    }
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run()
