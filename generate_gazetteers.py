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
from collections import Counter
import re

tokenizer = RegexpTokenizer("[\W+]", gaps=True)
stop = stopwords.words("english")

Entrez.email = "tsalo90@gmail.com"


def generate_metadata_gazetteers(label_file="/Users/salo/NBCLab/athena-data/processed_data/all_labels.csv",
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
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
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
    
    # Remove low-frequency title words
    tw_dict = Counter(title_word_gaz)
    for tw in tw_dict.keys():
        if tw_dict[tw] < 5 or tw.isdigit():
            del tw_dict[tw]
    tw_gaz = sorted(tw_dict.keys())
    
    # Remove low-frequency authors/years
    ay_dict = Counter(author_year_gaz)
    for ay in ay_dict.keys():
        if ay_dict[ay] < 5:
            del ay_dict[ay]
    ay_gaz = sorted(ay_dict.keys())
    
    # Remove low-frequency journals
    j_dict = Counter(journal_gaz)
    for j in j_dict.keys():
        if j_dict[j] < 5:
            del j_dict[j]
    j_gaz = sorted(j_dict.keys())
    
    # Remove duplicates
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


def extract_cogat(id_file, cogat_file, data_dir, out_file):
    """
    Creates feature table for Cognitive Atlas terms in text.
    Just a first pass.
    """
    # Read in instances
    id_df = pd.read_csv(id_file)
    pmids = id_df["pmid"].astype(str).tolist()
    
    # Read in features
    cogat_df = pd.read_csv(cogat_file)
    unique_terms = sorted(cogat_df["id"].unique().tolist())

    # Count    
    result_array = np.zeros((len(pmids), len(unique_terms)))
    for i, pmid in enumerate(pmids):
        data_file = os.path.join(data_dir, pmid+".txt")
        if not os.path.isfile(data_file):
            data_file = "/Users/salo/NBCLab/athena-data/testData/executiveData/{0}.txt".format(pmid)
        with open(data_file, "r") as fo:
            text = fo.read()
    
        for row in cogat_df.index:
            term = cogat_df["term"].iloc[row]
            term_id = cogat_df["id"].iloc[row]
            col_idx = unique_terms.index(term_id)
            if term in text:  # To be replaced with more advanced extraction method
                text.replace(term, term_id)
                count = len(re.findall(term, text))
                result_array[i, col_idx] += count

    # Create and save output
    out_df = pd.DataFrame(columns=unique_terms, index=pmids, data=result_array)
    out_df.index.name = "pmid"
    out_df.to_csv(out_file)


def extract_kw(id_file, kw_file, data_dir, out_file):
    """
    Creates feature table for keyword terms in text.
    Just a first pass.
    """
    # Read in instances
    id_df = pd.read_csv(id_file)
    pmids = id_df["pmid"].astype(str).tolist()
    
    # Read in features
    with open(kw_file, "rb") as fo:
        reader = csv.reader(fo, delimiter="\n")
        kw_list = list(reader)
    kw_list = sorted([item for row in kw_list for item in row])
    
    # Count
    result_array = np.zeros((len(pmids), len(kw_list)))
    for i, pmid in enumerate(pmids):
        data_file = os.path.join(data_dir, pmid+".txt")
        with open(data_file, "r") as fo:
            text = fo.read()
    
        for j, kw in enumerate(kw_list):
            if kw in text:  # To be replaced with more advanced extraction method
                result_array[i, j] += len(re.findall(kw, text))
    
    # Create and save output
    out_df = pd.DataFrame(columns=kw_list, index=pmids, data=result_array)
    out_df.index.name = "pmid"
    out_df.to_csv(out_file)


def extract_ay(list_of_pmids, ay_gaz, out_file):
    """
    Creates feature table for authors and year of publication.
    """
    column_names = ["pmid"] + ay_gaz
    df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
    df["pmid"] = list_of_pmids
    for pmid in df["pmid"]:
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
        ays = [author.lower() for author in record["AU"]]
        ays += [record["DP"][:4]]
        for ay in ays:
            if ay in ay_gaz:
                df[ay].loc[df["pmid"]==pmid] += 1
    df.to_csv(out_file, index=False)


def extract_j(list_of_pmids, j_gaz, out_file):
    """
    Creates feature table for journal of publication.
    """
    column_names = ["pmid"] + j_gaz
    df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
    df["pmid"] = list_of_pmids
    for pmid in df["pmid"]:
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
        js = [journal.lower() for journal in record["TA"]]
        for j in js:
            if j in j_gaz:
                df[j].loc[df["pmid"]==pmid] += 1
    df.to_csv(out_file, index=False)


def extract_tw(list_of_pmids, tw_gaz, out_file):
    """
    Creates feature table for words in article title.
    
    The tokenizer is definitely insufficient at the moment.
    """
    column_names = ["pmid"] + tw_gaz
    df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
    df["pmid"] = list_of_pmids
    for pmid in df["pmid"]:
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
        
        title = record["TI"]
        title_words = tokenizer.tokenize(title)
        title_words = [word.lower() for word in title_words if word.lower() not in stop]
        for tw in title_words:
            if tw in tw_gaz:
                df[tw].loc[df["pmid"]==pmid] += 1
    df.to_csv(out_file, index=False)


def extract_all(label_dir="/Users/salo/NBCLab/athena-data/processed_data/",
                gaz_dir="/Users/salo/NBCLab/athena-data/gazetteers/"):
    """
    Calls each of the feature-specific count functions to generate three
    feature count files. Keywords will be extracted from the articles' texts,
    not from their metadata.
    """    
    gaz_dict = {"ay": extract_ay,
                "j": extract_j,
                "tw": extract_tw}
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
    generate_metadata_gazetteers(os.path.join(sys.argv[1], "all_labels.csv"), sys.argv[2])
    extract_all(sys.argv[1], sys.argv[2])
