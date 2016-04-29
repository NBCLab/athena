# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 00:07:14 2016

Functions to generate count files for all Athena features, as well as to apply
weights to CogAt feature counts.

Inputs:
- IDs
- Text data
- Gazetteers

Outputs:
- Count files

@author: salo
"""

import os
import sys
import csv
import re
import copy
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

with open("misc/onix_stopwords.txt") as f:
    stop = f.read().split()

tokenizer = RegexpTokenizer("[\W+]", gaps=True)


def read_gazetteer(gazetteer_file):
    """
    Reads file into list.
    """
    with open(gazetteer_file, "rb") as fo:
        reader = csv.reader(fo, delimiter="\n")
        gazetteer = list(reader)
    gazetteer = [item for row in gazetteer for item in row]
    return gazetteer


def extract_nbow(pmids, gazetteer_file, count_file, text_dir):
    """
    Creates feature table for naive bag of words feature from stemmed text.
    """
    # Read in features
    gazetteer = read_gazetteer(gazetteer_file)
    
    # Count
    text_list = [[] for _ in pmids]
    for i, pmid in enumerate(pmids):
        file_ = os.path.join(text_dir, pmid+".txt")
        with open(file_, "r") as fo:
            text = fo.read()
            text_list[i] = text
    
    tfidf = TfidfVectorizer(vocabulary=gazetteer)
    result_array = tfidf.fit_transform(text_list).toarray()
    
    # Normalize matrix
    result_array = result_array / result_array.sum(axis=1)[:, None]
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=result_array)
    count_df.index.name = "pmid"
    count_df = count_df.fillna(0)
    count_df.to_csv(count_file)


def extract_features(data_dir="/home/data/nbc/athena/v1.1-data/"):
    """
    Calls each of the feature-specific count functions to generate three
    feature count files. Keywords will be extracted from the articles' texts,
    not from their metadata.
    """    
    datasets = ["train", "test"]
    for text_type in ["full", "combined"]:
        type_dir = os.path.join(data_dir, text_type)
        gazetteers_dir = os.path.join(type_dir, "gazetteers/")
        label_dir = os.path.join(type_dir, "labels/")
        feature_dir = os.path.join(type_dir, "features/")
        text_dir = os.path.join(data_dir, "text/", "stemmed_"+text_type)

        for dataset in datasets:
            label_file = os.path.join(label_dir, dataset+".csv")
            df = pd.read_csv(label_file)
            pmids = df["pmid"].astype(str).tolist()
    
            # nbow and references
            gazetteer_file = os.path.join(gazetteers_dir, "nbow.csv")
            count_file = os.path.join(feature_dir, "{0}_nbow.csv".format(dataset))
            extract_nbow(pmids, gazetteer_file, count_file, stemtext_dir)
            print("Completed {0} nbow".format(dataset))


if __name__ == "__main__":
    extract_features(sys.argv[1])
