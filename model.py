import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import joblib




df=pd.read_csv("test.csv",usecols=["text","sentiment"], encoding='latin1')
# print(df.head())


# Filling missing sentiment with mode would bias the model toward the most frequent class.
# Filling missing text with a placeholder adds fake text, which confuses NLP models.

df.dropna(inplace=True)
# print(df.drop_duplicates(inplace=True))
# print(df.duplicated().sum())



# now make text cleaning function 

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


df["text"]=df["text"].apply(text_cleaning)

df["sentiment"]=df["sentiment"].map({"positive":1,"negative":2,"neutral":0})


# now do train test split

X=df["text"]
Y=df["sentiment"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)



vectorizer=TfidfVectorizer()

vectorizer.fit(X_train)


X_train_vector=vectorizer.transform(X_train)
X_test_vector=vectorizer.transform(X_test)


    
lr=RandomForestClassifier(random_state=42)


lr.fit(X_train_vector,Y_train)

y_predict=lr.predict(X_test_vector)


joblib.dump(vectorizer,"tfidf vectorizer.pkl")
joblib.dump(lr,"model.pkl")



print(f"Confusion matrix is: ",confusion_matrix(Y_test,y_predict))
print(f"Classification report is: ",classification_report(Y_test,y_predict))
print(f"accuracy score is: ",accuracy_score(Y_test,y_predict))




# random forest result
# Confusion matrix is:  [[214  37  35]
#  [ 61 147   6]
#  [ 78  18 111]]
# Classification report is:                precision    recall  f1-score   support

#            0       0.61      0.75      0.67       286
#            1       0.73      0.69      0.71       214
#            2       0.73      0.54      0.62       207

#     accuracy                           0.67       707
#    macro avg       0.69      0.66      0.66       707
# weighted avg       0.68      0.67      0.67       707

# accuracy score is:  0.6676096181046676



