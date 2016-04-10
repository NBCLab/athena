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
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tokenize

tokenizer = RegexpTokenizer("[\W+]", gaps=True)
stop = stopwords.words("english")

Entrez.email = "tsalo90@gmail.com"


def generate_gazetteers(label_file="/Users/salo/NBCLab/athena-data/labels/full.csv",
                        gaz_dir="/Users/salo/NBCLab/athena-data/gazetteers/",
                        text_dir="/Users/salo/NBCLab/athena-data/text/"):
    """
    Creates list of unique terms for four gazetteers derived from metadata
    available through PubMed:
        - Authors and year of publication
        - Journal of publication
        - Words in title (not including stopwords)
        - Author-generated keywords (if available on PubMed)
            - This includes multiword expressions.
    """
    combined_text_dir = os.path.join(text_dir, "combined/")
    full_text_dir = os.path.join(text_dir, "full/")
    
    df = pd.read_csv(label_file)
    pmids = df["pmid"].astype(str).tolist()
    
    cogat_gaz, cogat_rels = generate_cogat_gazetteer()
    nbow_gaz = generate_nbow_gazetteer(pmids, combined_text_dir)
    references_gaz = generate_references_gazetteer(pmids, full_text_dir)
    metadata_gazs = generate_metadata_gazetteers(pmids)
    
    authoryear_gaz, journal_gaz, keyword_gaz, titleword_gaz = metadata_gazs
    
    save_gaz(cogat_gaz, gaz_dir, "cogat")
    save_gaz(nbow_gaz, gaz_dir, "nbow")
    save_gaz(references_gaz, gaz_dir, "references")
    save_gaz(authoryear_gaz, gaz_dir, "authoryear")
    save_gaz(journal_gaz, gaz_dir, "journal")
    save_gaz(titleword_gaz, gaz_dir, "titleword")
    save_gaz(keyword_gaz, gaz_dir, "keyword")


def generate_cogat_gazetteer():
    pass


def generate_nbow_gazetteer(pmids, text_dir):
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
    pass


def generate_metadata_gazetteers(pmids):
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
    tw_dict = Counter(titleword_gaz)
    for tw in tw_dict.keys():
        if tw_dict[tw] < 5 or tw.isdigit():
            del tw_dict[tw]
    titleword_gaz = sorted(tw_dict.keys())
    
    # Remove low-frequency authors/years
    ay_dict = Counter(authoryear_gaz)
    for ay in ay_dict.keys():
        if ay_dict[ay] < 5:
            del ay_dict[ay]
    authoryear_gaz = sorted(ay_dict.keys())
    
    # Remove low-frequency journals
    j_dict = Counter(journal_gaz)
    for j in j_dict.keys():
        if j_dict[j] < 5:
            del j_dict[j]
    journal_gaz = sorted(j_dict.keys())
    
    # Remove duplicate keywords
    keyword_gaz = sorted(list(set(keyword_gaz)) + titleword_gaz)
    return [authoryear_gaz, journal_gaz, keyword_gaz, titleword_gaz]
    

def save_gaz(gaz_list, gaz_dir, feature_name):
    gaz_file = os.path.join(gaz_dir, feature_name+".txt")
    with open(gaz_file, "w") as fo:
        writer = csv.writer(fo, lineterminator="\n")
        for att in gaz_list:
            writer.writerow([att])


if __name__ == "__main__":
    generate_gazetteers(os.path.join(sys.argv[1], "full.csv"), sys.argv[2])
