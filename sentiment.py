import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _wordnet_part_of_speech(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


@lru_cache(maxsize=1)
def _english_stop_words() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def clean_text(text: str) -> str:
    tokens = word_tokenize(text.lower())
    stop_words = _english_stop_words()
    without_stop_words = [word for word in tokens if word not in stop_words]
    without_punctuation = [
        word for word in without_stop_words if word not in string.punctuation
    ]

    cleaned = " ".join(without_punctuation)
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)
    cleaned = re.sub(r"<.*?>", "", cleaned)
    cleaned = re.sub(r"http\s*//\S+|www\S+", " ", cleaned)

    tagged_tokens = nltk.pos_tag(word_tokenize(cleaned))
    lemmatizer = _lemmatizer()
    lemmatized = [
        lemmatizer.lemmatize(word, pos=_wordnet_part_of_speech(tag))
        for word, tag in tagged_tokens
    ]
    return " ".join(lemmatized)
