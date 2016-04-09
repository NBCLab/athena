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
from Bio import Entrez
from Bio import Medline
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
import sys
import csv
import re
import copy

tokenizer = RegexpTokenizer("[\W+]", gaps=True)
stop = stopwords.words("english")

Entrez.email = "tsalo90@gmail.com"


def extract_features():
    pass


def read_gazetteer(gazetteer_file):
    """
    Reads file into list.
    """
    with open(gazetteer_file, "rb") as fo:
        reader = csv.reader(fo, delimiter="\n")
        gazetteer = list(reader)
    gazetteer = [item for row in gazetteer for item in row]
    return gazetteer


def extract_nbow(pmids, gazetteer_file, count_file, data_dir):
    """
    Creates feature table for naive bag of words feature from text.
    Just a first pass.
    """
    # Read in features
    gazetteer = read_gazetteer(gazetteer_file)
    
    # Count
    result_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        data_file = os.path.join(data_dir, pmid+".txt")
        with open(data_file, "r") as fo:
            text = fo.read()
    
        for j, word in enumerate(gazetteer):
            if word in text:  # To be replaced with more advanced extraction method
                result_array[i, j] += len(re.findall(word, text))
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=result_array)
    count_df.index.name = "pmid"
    count_df.to_csv(count_file)


def extract_references(pmids, gazetteer_file, count_file, data_dir):
    """
    """
    pass


def extract_cogat(pmids, gazetteer_file, count_file, data_dir):
    """
    Creates feature table for Cognitive Atlas terms from text.
    Just a first pass.
    """
    # Read in features
    cogat_df = pd.read_csv(gazetteer_file)
    gazetteer = sorted(cogat_df["id"].unique().tolist())

    # Count    
    count_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        data_file = os.path.join(data_dir, pmid+".txt")
        with open(data_file, "r") as fo:
            text = fo.read()
    
        for row in cogat_df.index:
            term = cogat_df["term"].iloc[row]
            term_id = cogat_df["id"].iloc[row]
            col_idx = gazetteer.index(term_id)
            if term in text:  # To be replaced with more advanced extraction method
                text.replace(term, term_id)
                count = len(re.findall(term, text))
                count_array[i, col_idx] += count

    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=count_array)
    count_df.index.name = "pmid"
    count_df.to_csv(count_file)


def apply_weights(input_df, weight_df):
    """
    Apply weights once. I (TS) doubt this will be used.
    """
    weight_df = weight_df.reindex_axis(sorted(weight_df.columns), axis=1).sort()
    input_df = input_df.reindex_axis(sorted(input_df.columns), axis=1)
    weighted_df = input_df.dot(weight_df)
    return weighted_df


def apply_weights_recursively(input_df, weight_dfs=None, weighting_scheme="ws2"):
    """
    First pass at trying to apply parent- and child-directed weights all the
    way to their ends. Sideways weights are only applied once.

    input_df:           A DataFrame with observed feature counts.

    weight_dfs:         A list of DataFrames corresponding to upward-,
                        downward-, and side-directed relationships. Either
                        weight_dfs or rel_df must be defined.

    rel_df:             A DataFrame of existing relationships from the
                        Cognitive Atlas. Will be used with weighting_scheme to
                        create weight_dfs. Either weight_dfs or rel_df must be
                        defined.
    weighting_scheme:   The weighting scheme. Must match a set of keys from
                        get_weights.
    """
    if type(weight_dfs) == list:
        weights_up = weight_dfs[0]
        weights_down = weight_dfs[1]
        weights_side = weight_dfs[2]
    else:
        raise Exception("Neither weight_dfs nor rel_df defined.")

    weights_up = weights_up.reindex_axis(sorted(weights_up.columns), axis=1).sort()
    weights_down = weights_down.reindex_axis(sorted(weights_down.columns), axis=1).sort()
    weights_side = weights_side.reindex_axis(sorted(weights_side.columns), axis=1).sort()
    input_df = input_df.reindex_axis(sorted(input_df.columns), axis=1)
    
    zero_df = copy.deepcopy(input_df)
    zero_df[zero_df>-1]=0

    # Apply vertical relationship weights until relationships are exhausted
    counter = 0
    weighted_up = input_df.dot(weights_up)
    input_df += weighted_up
    while not weighted_up.equals(zero_df):
        weighted_up = weighted_up.dot(weights_up)
        input_df += weighted_up
        counter += 1
        if counter > (1700):
            break

    counter = 0
    weighted_down = input_df.dot(weights_down)
    input_df += weighted_down
    while not weighted_down.equals(zero_df):
        weighted_down = weighted_down.dot(weights_down)
        input_df += weighted_down
        counter += 1
        if counter > (1700):
            break

    # Apply horizontal relationship weights once
    weighted_side = input_df.dot(weights_side)
    input_df += weighted_side
    return input_df


def extract_keywords(pmids, gazetteer_file, count_file, data_dir):
    """
    Creates feature table for keyword terms from text.
    Just a first pass.
    """
    # Read in features
    gazetteer = read_gazetteer(gazetteer_file)
    
    # Count
    result_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        data_file = os.path.join(data_dir, pmid+".txt")
        with open(data_file, "r") as fo:
            text = fo.read()
    
        for j, keyword in enumerate(gazetteer):
            if keyword in text:  # To be replaced with more advanced extraction method
                result_array[i, j] += len(re.findall(keyword, text))
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=result_array)
    count_df.index.name = "pmid"
    count_df.to_csv(count_file)


def extract_authoryear(pmids, gazetteer_file, count_file):
    """
    Creates feature table for authors and year of publication.
    """
    # Read in features
    gazetteer = read_gazetteer(gazetteer_file)

    # Count    
    result_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
        authoryears = [author.lower() for author in record["AU"]]
        authoryears += [record["DP"][:4]]
        for j, authoryear in enumerate(gazetteer):
            if authoryear in authoryears:
                result_array[i, j] += 1
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=result_array)
    count_df.index.name = "pmid"
    count_df.to_csv(count_file)


def extract_journal(pmids, gazetteer_file, count_file):
    """
    Creates feature table for journal of publication.
    """
    # Read in features
    gazetteer = read_gazetteer(gazetteer_file)
    
    # Count
    result_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
        journals = [journal.lower() for journal in record["TA"]]
        for j, journal in enumerate(gazetteer):
            if journal in journals:
                result_array[i, j] += 1
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=result_array)
    count_df.index.name = "pmid"
    count_df.to_csv(count_file)


def extract_titlewords(pmids, gazetteer_file, count_file):
    """
    Creates feature table for words in article title.
    """
    # Read in features
    gazetteer = read_gazetteer(gazetteer_file)
    
    # Count
    result_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
        title = record["TI"]
        titlewords = tokenizer.tokenize(title)
        titlewords = [word.lower() for word in titlewords if word.lower() not in stop]
        for j, titleword in enumerate(gazetteer):
            if titleword in titlewords:
                result_array[i, j] += 1
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=result_array)
    count_df.index.name = "pmid"
    count_df.to_csv(count_file)


def extract_all(data_dir="/Users/salo/NBCLab/athena-data/"):
    """
    Calls each of the feature-specific count functions to generate three
    feature count files. Keywords will be extracted from the articles' texts,
    not from their metadata.
    """    
    feature_dict = {"authoryear": extract_authoryear,
                    "journal": extract_journal,
                    "titlewords": extract_titlewords}
    datasets = ["train", "test"]

    gazetteers_dir = os.path.join(data_dir, "gazetteers/")
    label_dir = os.path.join(data_dir, "labels/")
    feature_dir = os.path.join(data_dir, "features/")

    for feature in feature_dict.keys():        
        gazetteer = read_gazetteer(gazetteers_dir, feature)
        
        for dataset in datasets:
            label_file = os.path.join(label_dir, dataset+".csv")
            df = pd.read_csv(label_file)
            pmids = df["pmid"].astype(str).tolist()

            feature_dict[feature](pmids, gazetteer, os.path.join(feature_dir, "{0}_{1}.csv".format(dataset, feature)))
