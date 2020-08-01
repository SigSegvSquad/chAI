import pickle
import numpy as np
import nltk
import random
import json
from keras.models import load_model
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
words= pickle.load(open('words.pkl', 'rb'))
classes= pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open("intents.json",encoding="utf8"))

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

sentence="whatsup"
context="0"
p = bow(sentence, words, context, show_details=False)
res = model.predict(np.array([p]))[0]
ERROR_THRESHOLD = 0.25
results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
return_list = []
for r in results:
  return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

tag = return_list[0]['intent']
list_of_intents = intents['intents']
for i in list_of_intents:
  if(i['tag']== tag):
    result = random.choice(i['responses'])
    break

print(result)

