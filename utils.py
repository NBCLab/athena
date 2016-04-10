# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:52:44 2016

@author: salo
"""

from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
stop = stopwords.words("english")


def stem_tokens(tokens, stemmer):
    """
    http://stackoverflow.com/questions/26126442/combining-text-stemming-and-
    removal-of-punctuation-in-nltk-and-scikit-learn
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    """
    http://stackoverflow.com/questions/26126442/combining-text-stemming-and-
    removal-of-punctuation-in-nltk-and-scikit-learn
    """
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
