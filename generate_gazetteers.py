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
import sys

tokenizer = RegexpTokenizer("[\s:\.]+", gaps=True)
stop = stopwords.words("english")

Entrez.email = "tsalo90@gmail.com"


def generate_metadata_gazeteers(label_file="/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv",
                                gaz_file = "/Users/salo/NBCLab/athena-data/gazetteers/gazetteers.pkl"):
    """
    Creates list of unique terms for four gazetteers derived from metadata
    available through PubMed:
        - Authors and year of publication
        - Journal of publication
        - Words in title (not including stopwords)
        - Author-generated keywords (if available on PubMed)
            - This includes multiword expressions.
    """
    df = pd.read_csv(label_file)
    pmids = df["pmid"].astype(str).tolist()
    
    author_year_gaz = []
    journal_gaz = []
    title_word_gaz = []
    keyword_gaz = []
    
    for pmid in pmids:
        h = Entrez.efetch(db='pubmed', id=pmid, rettype='medline', retmode='text')
        record = list(Medline.parse(h))[0]
        author_year_gaz += [author.replace(".", "").lower() for author in record["AU"]]
        author_year_gaz += [record["DP"][:4]]
        title = record["TI"]
        title_words = tokenizer.tokenize(title)
        title_words = [word.lower() for word in title_words if word.lower() not in stop]
        title_word_gaz += title_words
        journal_gaz += [record["TA"].lower()]
        if "OT" in record.keys():
            keywords = [keyword.lower() for keyword in record["OT"]]
            keyword_gaz += keywords
    
    # Remove duplicates
    ay_gaz = sorted(list(set(author_year_gaz)))
    j_gaz = sorted(list(set(journal_gaz)))
    tw_gaz = sorted(list(set(title_word_gaz)))
    k_gaz = sorted(list(set(keyword_gaz)))
    
    with open(gaz_file, "wb") as fo:
        pickle.dump([ay_gaz, j_gaz, tw_gaz, k_gaz], fo)


def count_ay_metadata(list_of_pmids, ay_gaz, out_file):
    """
    Creates feature table for authors and year of publication.
    """
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
    """
    Creates feature table for journal of publication.
    """
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
    Creates feature table for words in article title.
    
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


def count_gazs(label_dir="/Users/salo/NBCLab/athena-data/processed_data/",
               gaz_file="/Users/salo/NBCLab/athena-data/gazetteers/gazetteers.pkl"):
    """
    Calls each of the feature-specific count functions to generate three
    feature count files. Keywords will be extracted from the articles' texts,
    not from their metadata.
    """  
    with open(gaz_file, "rb") as fo:
        ay_gaz, j_gaz, tw_gaz, k_gaz = pickle.load(fo)
    
    for dataset in ["train", "test"]:
        filename = "{0}_labels.csv".format(dataset)
        label_file = os.path.join(label_dir, filename)
        df = pd.read_csv(label_file)
        pmids = df["pmid"].astype(str).tolist()

        count_ay_metadata(pmids, ay_gaz, os.path.join(label_dir, "{0}_features_ay.csv".format(dataset)))
        count_j_metadata(pmids, j_gaz, os.path.join(label_dir, "{0}_features_j.csv".format(dataset)))
        count_tw_metadata(pmids, tw_gaz,os.path.join(label_dir, "{0}_features_tw.csv".format(dataset)))


if __name__ == "__main__":
    count_gazs(sys.argv[1], sys.argv[2])
