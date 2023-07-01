
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


cv = pickle.load(open('/content/vectorizer.pkl','rb'))
model = pickle.load(open('/content/model.pkl','rb'))

ps = PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

st.title("Ask Your Query To The Bot")

input_sms = st.text_area("Enter the message")

if st.button('Answer'):

    transformed_sms = transform_text(input_sms)
  
    vector_input = cv.transform([transformed_sms])
    
    result = model.predict(vector_input)[0]

    st.write(result)
    
