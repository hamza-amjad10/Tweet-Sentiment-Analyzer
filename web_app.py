import pandas as pd
import numpy as np
import streamlit as st
import joblib
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import joblib


model=joblib.load("model.pkl")
vectorizer=joblib.load("tfidf vectorizer.pkl")

st.title("Tweet Sentiment App")

tweet=st.text_area("Enter your tweet here! ", placeholder="I am really frustrated with this service. It keeps crashing and nothing works.",height="stretch")




def text_cleaning(text):
   main_text=text.lower()
   tokens=word_tokenize(main_text)
   # nltk.download("stopwords")
   stopwards_check=set(stopwords.words("english"))
   remove_stopwords=[w for w in tokens if w not in stopwards_check]
   # print(remove_stopwords)
   remove_punctuations=[w for w in remove_stopwords if w not in string.punctuation]
   # print(remove_punctuations)
   remove_punctuation=" ".join(remove_punctuations)
   remove_numbers=re.sub('\d+'," ",remove_punctuation)
   # print(remove_numbers)
   remove_spaces=re.sub('\s+'," ",remove_numbers) 
   # print(remove_spaces)
   remove_specials=re.sub(r'[^\x00-\x7F]+'," ",remove_spaces)
   # print(remove_specials)
   remove_tags=re.sub('<.*?>',"",remove_specials)
   # print(remove_tags)
   clean_text = re.sub(r'http\s*//\S+|www\S+', ' ', remove_tags)
   lemmi=WordNetLemmatizer()
   # nltk.download("averaged_preceptron_tagger")
   # nltk.download('wordnet')
   original_tokens=word_tokenize(clean_text)
   pos_tags=nltk.pos_tag(original_tokens)
   def convert_wordnet_pos(words):
         if words.startswith("J"):
          return wordnet.ADJ
         elif words.startswith("V"):
          return wordnet.VERB
         elif words.startswith("N"):
          return wordnet.NOUN
         elif words.startswith("R"):
          return wordnet.ADV
         else:
          return wordnet.NOUN
      
   new_list = []
 
   for word,pos in pos_tags:
       pos_single_tag=convert_wordnet_pos(pos)
       lematized_words=lemmi.lemmatize(word,pos=pos_single_tag)
       new_list.append(lematized_words)
 
   return " ".join(new_list)

clean_tweet=text_cleaning(tweet)


if st.button("predict"):
    text_vector=vectorizer.transform([clean_tweet])
    prediction=model.predict(text_vector)
    proba = model.predict_proba(text_vector)
    confidence = np.max(proba)  
    st.progress(confidence)
    st.write(f"Confidence: {confidence*100:.2f}%")


    if prediction[0]==0:
        sentiment="neutral"
    elif prediction[0]==1:
        sentiment="positive"
    else:
        sentiment="negative"
    
    if sentiment=="neutral":
     st.info(f"Model predict the tweet is: {sentiment} 😶")
    elif sentiment=="positive":
     st.success(f"Model predict the tweet is: {sentiment} 🙂")
    else:
     st.error(f"Model predict the tweet is: {sentiment} 😞")