# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:23:56 2016
Generate keyword, author/year, journal, and title-word gazetteers.
Also provide counts for author/year, journal, and title-word features.
@author: salo
"""

from Bio import Entrez
from Bio import Medline
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pickle
import numpy as np
import os

tokenizer = RegexpTokenizer("[\s:\.]+", gaps=True)
stop = stopwords.words("english")

Entrez.email = "tsalo90@gmail.com"

def generate_keyword_gazetteer(training_pmids):
    keyword_gaz = []
    
    for pmid in training_pmids:
        h = Entrez.efetch(db='pubmed', id=pmid, rettype='medline', retmode='text')
        record = list(Medline.parse(h))[0]
        if "OT" in record.keys():
            keywords = record["OT"]
            keyword_gaz += keywords
    
    keyword_gaz = [keyword.lower() for keyword in keyword_gaz]
    keyword_gaz = sorted(list(set(keyword_gaz)))
    return keyword_gaz


def generate_metadata_gazeteers(training_pmids):
    author_year_gaz = []
    journal_gaz = []
    title_word_gaz = []
    
    for pmid in training_pmids:
        h = Entrez.efetch(db='pubmed', id=pmid, rettype='medline', retmode='text')
        record = list(Medline.parse(h))[0]
        author_year_gaz += record["AU"]
        author_year_gaz += [record["DP"][:4]]
        title = record["TI"]
        title_words = tokenizer.tokenize(title)
        title_words = [word.lower() for word in title_words if word.lower() not in stop]
        title_word_gaz += title_words
        journal_gaz += [record["TA"]]
    author_year_gaz = [ay.replace(".", "").lower() for ay in author_year_gaz]
    author_year_gaz = sorted(list(set(author_year_gaz)))
    journal_gaz = [j.lower() for j in journal_gaz]
    journal_gaz = sorted(list(set(journal_gaz)))
    title_word_gaz = sorted(list(set(title_word_gaz)))
    
    return author_year_gaz, journal_gaz, title_word_gaz


def count_ay_metadata(list_of_pmids, ay_gaz, out_file):
    column_names = ["pmid"] + ay_gaz
    df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
    df["pmid"] = list_of_pmids
    for pmid in df["pmid"]:
        h = Entrez.efetch(db='pubmed', id=pmid, rettype='medline', retmode='text')
        record = list(Medline.parse(h))[0]
        ays = [author.lower() for author in record["AU"]]
        ays += [record["DP"][:4]]
        for ay in ays:
            if ay in ay_gaz:
                df[ay].loc[df["pmid"]==pmid] += 1
    df.to_csv(out_file, index=False)


def count_j_metadata(list_of_pmids, j_gaz, out_file):
    column_names = ["pmid"] + j_gaz
    df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
    df["pmid"] = list_of_pmids
    for pmid in df["pmid"]:
        h = Entrez.efetch(db='pubmed', id=pmid, rettype='medline', retmode='text')
        record = list(Medline.parse(h))[0]
        js = [journal.lower() for journal in record["TA"]]
        for j in js:
            if j in j_gaz:
                df[j].loc[df["pmid"]==pmid] += 1
    df.to_csv(out_file, index=False)


def count_tw_metadata(list_of_pmids, tw_gaz, out_file):
    """
    The tokenizer is definitely insufficient at the moment.
    """
    column_names = ["pmid"] + tw_gaz
    df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
    df["pmid"] = list_of_pmids
    for pmid in df["pmid"]:
        h = Entrez.efetch(db='pubmed', id=pmid, rettype='medline', retmode='text')
        record = list(Medline.parse(h))[0]
        
        title = record["TI"]
        title_words = tokenizer.tokenize(title)
        title_words = [word.lower() for word in title_words if word.lower() not in stop]
        for tw in title_words:
            if tw in tw_gaz:
                df[tw].loc[df["pmid"]==pmid] += 1
    df.to_csv(out_file, index=False)


def test():
    training_pmids = ["26500517", "26494483"]
    k_gaz = generate_keyword_gazetteer(training_pmids)
    ay_gaz, j_gaz, tw_gaz = generate_metadata_gazeteers(training_pmids)
    with open("gazetteers.pkl", "wb") as fo:
        pickle.dump([k_gaz, ay_gaz, j_gaz, tw_gaz], fo)
    return ay_gaz


def create_gazs(folder="/Users/salo/NBCLab/athena-data/meta_data/"):
    filenames = ["Face.csv", "Pain.csv", "Passive.csv", "Reward.csv",
                 "Semantic.csv", "Word.csv", "nBack.csv"]
    all_pmids = []
    for file_ in filenames:
        full_file = os.path.join(folder, file_)
        df = pd.read_csv(full_file, dtype=str)
        pmids = df["PubMed ID"].unique().tolist()
        all_pmids += pmids
    all_pmids = [pmid for pmid in all_pmids if pmid.isdigit()]
    all_pmids = sorted(list(set(all_pmids)))
    k_gaz = generate_keyword_gazetteer(all_pmids)
    ay_gaz, j_gaz, tw_gaz = generate_metadata_gazeteers(all_pmids)
    with open("/Users/salo/NBCLab/athena-data/gazetteers/gazetteers.pkl", "wb") as fo:
        pickle.dump([k_gaz, ay_gaz, j_gaz, tw_gaz], fo)


def count_gazs(folder="/Users/salo/NBCLab/athena-data/meta_data/"):
    filenames = ["Face.csv", "Pain.csv", "Passive.csv", "Reward.csv",
                 "Semantic.csv", "Word.csv", "nBack.csv"]
    all_pmids = []
    for file_ in filenames:
        full_file = os.path.join(folder, file_)
        df = pd.read_csv(full_file, dtype=str)
        pmids = df["PubMed ID"].unique().tolist()
        all_pmids += pmids
    all_pmids = [pmid for pmid in all_pmids if pmid.isdigit()]
    all_pmids = sorted(list(set(all_pmids)))
    
    with open("/Users/salo/NBCLab/athena-data/gazetteers/gazetteers.pkl", "rb") as fo:
        k_gaz, ay_gaz, j_gaz, tw_gaz = pickle.load(fo)
    
    count_ay_metadata(all_pmids, ay_gaz, "/Users/salo/NBCLab/athena-data/processed_data/train_features_ay.csv")
    count_j_metadata(all_pmids, j_gaz, "/Users/salo/NBCLab/athena-data/processed_data/train_features_j.csv")
    count_tw_metadata(all_pmids, tw_gaz, "/Users/salo/NBCLab/athena-data/processed_data/train_features_tw.csv")
