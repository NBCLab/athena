# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:23:56 2016
Generate keyword, author/year, journal, and title-word gazetteers.
@author: salo
"""

import os
import csv
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = RegexpTokenizer("[\W+]", gaps=True)

with open("../athena-data/misc/onix_stopwords.txt") as f:
    stop = f.read().split()


def generate_nbow_gazetteer(pmids, text_dir):
    """
    """
    text_list = [[] for _ in pmids]
    for i, pmid in enumerate(pmids):
        file_ = os.path.join(text_dir, pmid+".txt")
        with open(file_, "r") as fo:
            text = fo.read()
            text_list[i] = text
    
    tfidf = TfidfVectorizer(stop_words=stop,
                            sublinear_tf=True,
                            min_df=3)
    tfidf.fit(text_list)
    unicode_gaz = tfidf.vocabulary_.keys()
    nbow_gaz = [str(w) for w in unicode_gaz]
    return nbow_gaz


def save_gaz(gaz_list, gaz_dir, feature_name):
    """
    """
    gaz_file = os.path.join(gaz_dir, feature_name+".csv")
    with open(gaz_file, "w") as fo:
        writer = csv.writer(fo, lineterminator="\n")
        for att in gaz_list:
            writer.writerow([att])
