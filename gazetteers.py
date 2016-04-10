# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:23:56 2016
Generate keyword, author/year, journal, and title-word gazetteers.
@author: salo
"""

import os
import sys
import csv
from Bio import Entrez
from Bio import Medline
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tokenize

tokenizer = RegexpTokenizer("[\W+]", gaps=True)
stop = stopwords.words("english")

Entrez.email = "tsalo90@gmail.com"


def generate_nbow_gazetteer(pmids, text_dir):
    """
    """
    text_list = [[] for _ in pmids]
    for i, pmid in enumerate(pmids):
        file_ = os.path.join(text_dir, pmid+".txt")
        with open(file_, "r") as fo:
            text = fo.read()
            text_list[i] = text
    
    tfidf = TfidfVectorizer(tokenizer=tokenize,
                            stop_words=stop,
                            ngram_range=(1, 2))
    tfidf.fit(text_list)
    nbow_gaz = tfidf.get_feature_names()
    return nbow_gaz


def generate_references_gazetteer(pmids, text_dir):
    """
    """
    pass


def generate_metadata_gazetteers(pmids):
    """
    """
    authoryear_gaz = []
    journal_gaz = []
    keyword_gaz = []
    titleword_gaz = []
    
    for pmid in pmids:
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
        authoryear_gaz += [author.replace(".", "").lower() for author in record["AU"]]
        authoryear_gaz += [record["DP"][:4]]
        title = record["TI"]
        titlewords = tokenizer.tokenize(title)
        titlewords = [word.lower() for word in titlewords if word.lower() not in stop]
        titleword_gaz += titlewords
        journal_gaz += [record["TA"].lower()]
        if "OT" in record.keys():
            keywords = [keyword.lower() for keyword in record["OT"]]
            keyword_gaz += keywords
    
    # Remove low-frequency title words
    titleword_count = Counter(titleword_gaz)
    for tw in titleword_count.keys():
        if titleword_count[tw] < 5 or tw.isdigit():
            del titleword_count[tw]
    titleword_gaz = sorted(titleword_count.keys())
    
    # Remove low-frequency authors/years
    authoryear_count = Counter(authoryear_gaz)
    for ay in authoryear_count.keys():
        if authoryear_count[ay] < 5:
            del authoryear_count[ay]
    authoryear_gaz = sorted(authoryear_count.keys())
    
    # Remove low-frequency journals
    journal_count = Counter(journal_gaz)
    for j in journal_count.keys():
        if journal_count[j] < 5:
            del journal_count[j]
    journal_gaz = sorted(journal_count.keys())
    
    # Remove duplicate keywords
    keyword_gaz = sorted(list(set(keyword_gaz)) + titleword_gaz)
    return [authoryear_gaz, journal_gaz, keyword_gaz, titleword_gaz]
    

def save_gaz(gaz_list, gaz_dir, feature_name):
    """
    """
    gaz_file = os.path.join(gaz_dir, feature_name+".txt")
    with open(gaz_file, "w") as fo:
        writer = csv.writer(fo, lineterminator="\n")
        for att in gaz_list:
            writer.writerow([att])


if __name__ == "__main__":
    generate_gazetteers(os.path.join(sys.argv[1], "full.csv"), sys.argv[2])
