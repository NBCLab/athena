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
import numpy as np
import os
import sys
import csv

tokenizer = RegexpTokenizer("[\s:\.]+", gaps=True)
stop = stopwords.words("english")

Entrez.email = "tsalo90@gmail.com"


def generate_metadata_gazetteers(label_file="/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv",
                                 gaz_dir="/Users/salo/NBCLab/athena-data/gazetteers/"):
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
    
    save_gaz(ay_gaz, gaz_dir, "ay")
    save_gaz(j_gaz, gaz_dir, "j")
    save_gaz(tw_gaz, gaz_dir, "tw")
    save_gaz(k_gaz, gaz_dir, "k")


def save_gaz(gaz_list, gaz_dir, gaz_name):
    gaz_file = os.path.join(gaz_dir, gaz_name + "_gaz.txt")
    with open(gaz_file, "w") as fo:
        writer = csv.writer(fo, lineterminator="\n")
        for att in gaz_list:
            writer.writerow([att])    


def read_gaz(gaz_dir, gaz_name):
    gaz_file = os.path.join(gaz_dir, gaz_name + "_gaz.txt")
    with open(gaz_file, "rb") as fo:
        reader = csv.reader(fo, delimiter="\n")
        gaz_list = list(reader)
    gaz_list = [item for row in gaz_list for item in row]
    return gaz_list


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
               gaz_dir="/Users/salo/NBCLab/athena-data/gazetteers/"):
    """
    Calls each of the feature-specific count functions to generate three
    feature count files. Keywords will be extracted from the articles' texts,
    not from their metadata.
    """    
    gaz_dict = {"ay": count_ay_metadata,
                "j": count_j_metadata,
                "tw": count_tw_metadata}
    datasets = ["train", "test"]
                
    for gaz in gaz_dict.keys():        
        gaz_list = read_gaz(gaz_dir, gaz)
        
        for dataset in datasets:
            filename = "{0}_labels.csv".format(dataset)
            label_file = os.path.join(label_dir, filename)
            df = pd.read_csv(label_file)
            pmids = df["pmid"].astype(str).tolist()

            gaz_dict[gaz](pmids, gaz_list, os.path.join(label_dir, "{0}_features_{1}.csv".format(dataset, gaz)))


if __name__ == "__main__":
    generate_metadata_gazetteers(os.path.join(sys.argv[1], "train_labels.csv"), sys.argv[2])
    count_gazs(sys.argv[1], sys.argv[2])
