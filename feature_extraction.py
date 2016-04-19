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
from Bio import Entrez
from Bio import Medline
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tokenize

stop = stopwords.words("english")
tokenizer = RegexpTokenizer("[\W+]", gaps=True)

Entrez.email = "tsalo006@fiu.edu"


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
    
    tfidf = TfidfVectorizer(tokenizer=tokenize, vocabulary=gazetteer)
    result_array = tfidf.fit_transform(text_list).toarray()
    
    # Normalize matrix
    result_array = result_array / result_array.sum(axis=1)[:, None]
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=result_array)
    count_df.index.name = "pmid"
    count_df.to_csv(count_file)


def extract_references(pmids, gazetteer_file, count_file, text_dir):
    """
    Creates feature table for references feature from text.
    """
    pass


def extract_cogat(pmids, gazetteer_file, count_file, text_dir):
    """
    Creates feature table for Cognitive Atlas terms from full, unstemmed text.
    Just a first pass.
    """
    # Read in features
    cogat_df = pd.read_csv(gazetteer_file)
    gazetteer = sorted(cogat_df["id"].unique().tolist())

    # Count    
    count_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        text_file = os.path.join(text_dir, pmid+".txt")
        with open(text_file, "r") as fo:
            text = fo.read()
    
        for row in cogat_df.index:
            term = cogat_df["term"].iloc[row]
            words = term.split(" ")
            regex = "\\s*(\\(.*\\))?\\s*".join(words)
            pattern = re.compile(regex, re.MULTILINE|re.DOTALL)
            
            term_id = cogat_df["id"].iloc[row]
            col_idx = gazetteer.index(term_id)
            
            count = len(re.findall(pattern, text))
            count_array[i, col_idx] += count
            text = text.replace(term, term_id)

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
    
    if not (weight_df.columns == input_df.columns):
        raise Exception("Columns do not match between DataFrames!")

    weighted_df = input_df.dot(weight_df)
    return weighted_df


def apply_weights_recursively(input_df, weight_dfs, weighted_file):
    """
    First pass at trying to apply parent- and child-directed weights all the
    way to their ends. Sideways weights are only applied once.

    input_df:           A DataFrame with observed feature counts.

    weight_dfs:         A list of DataFrames corresponding to upward-,
                        downward-, and side-directed relationships. Either
                        weight_dfs or rel_df must be defined.
    """
    weights_up = weight_dfs[0].reindex_axis(sorted(weight_dfs[0].columns), axis=1).sort()
    weights_down = weight_dfs[1].reindex_axis(sorted(weight_dfs[1].columns), axis=1).sort()
    weights_side = weight_dfs[2].reindex_axis(sorted(weight_dfs[2].columns), axis=1).sort()

    # Add categories
    not_in_gaz = list(set(weights_up.columns.tolist()) - set(input_df.columns.tolist()))
    add = [term for term in not_in_gaz if term.startswith("ctp")]
    for term in add:
        input_df[term] = 0

    input_df = input_df.reindex_axis(sorted(input_df.columns), axis=1)
    n_features = input_df.shape[1]

    if not (weights_up.columns.tolist() == weights_down.columns.tolist() == weights_side.columns.tolist() == input_df.columns.tolist()):
        raise Exception("Columns do not match across DataFrames!")
    
    count_df = copy.deepcopy(input_df)
    zero_df = copy.deepcopy(input_df)
    zero_df[zero_df>-1]=0

    # Apply vertical relationship weights until relationships are exhausted
    counter = 0
    weighted_up = input_df.dot(weights_up)
    count_df += weighted_up
    while not weighted_up.equals(zero_df):
        weighted_up = weighted_up.dot(weights_up)
        count_df += weighted_up
        counter += 1
        if counter == (n_features-1):
            break

    counter = 0
    weighted_down = input_df.dot(weights_down)
    count_df += weighted_down
    while not weighted_down.equals(zero_df):
        weighted_down = weighted_down.dot(weights_down)
        count_df += weighted_down
        counter += 1
        if counter == (n_features-1):
            break

    # Apply horizontal relationship weights once
    weighted_side = input_df.dot(weights_side)
    count_df += weighted_side
    
    # Normalize data
    weighted_df = count_df.div(count_df.sum(axis=1), axis=0, fill_value=0)
    weighted_df.to_csv(weighted_file)


def extract_keywords(pmids, gazetteer_file, count_file, text_dir):
    """
    Creates feature table for keyword terms from full text.
    Just a first pass.
    """
    # Read in features
    gazetteer = read_gazetteer(gazetteer_file)
    
    # Count
    result_array = np.zeros((len(pmids), len(gazetteer)))
    for i, pmid in enumerate(pmids):
        text_file = os.path.join(text_dir, pmid+".txt")
        with open(text_file, "r") as fo:
            text = fo.read()
    
        for j, keyword in enumerate(gazetteer):
            # To be replaced with more advanced extraction method
            result_array[i, j] += len(re.findall(keyword, text))

    # Normalize matrix
    result_array = result_array / result_array.sum(axis=1)[:, None]    
    
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

    # Normalize matrix
    result_array = result_array / result_array.sum(axis=1)[:, None]

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
    
    # Normalize matrix
    result_array = result_array / result_array.sum(axis=1)[:, None]
    
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
    
    # Normalize matrix
    result_array = result_array / result_array.sum(axis=1)[:, None]
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer, index=pmids, data=result_array)
    count_df.index.name = "pmid"
    count_df.to_csv(count_file)


def extract_features(data_dir="/home/data/nbc/athena/athena-data/"):
    """
    Calls each of the feature-specific count functions to generate three
    feature count files. Keywords will be extracted from the articles' texts,
    not from their metadata.
    """    
    metadata_features = {"authoryear": extract_authoryear,
                         "journal": extract_journal,
                         "titlewords": extract_titlewords,
                         }
    fulltext_features = {"cogat": extract_cogat,
                         "keywords": extract_keywords,
                         }
    datasets = ["train", "test"]

    gazetteers_dir = os.path.join(data_dir, "gazetteers/")
    label_dir = os.path.join(data_dir, "labels/")
    feature_dir = os.path.join(data_dir, "features/")
    fulltext_dir = os.path.join(data_dir, "text/full/")
#    stemtext_dir = os.path.join(data_dir, "text/full_stemmed/")
#    reftext_dir = os.path.join(data_dir, "text/references/")

    for dataset in datasets:
        label_file = os.path.join(label_dir, dataset+".csv")
        df = pd.read_csv(label_file)
        pmids = df["pmid"].astype(str).tolist()
        
        for feature in metadata_features:
            gazetteer_file = os.path.join(gazetteers_dir, feature+".csv")
            count_file = os.path.join(feature_dir, "{0}_{1}.csv".format(dataset, feature))
            metadata_features[feature](pmids, gazetteer_file, count_file)
            print("Completed {0} {1}".format(dataset, feature))
        
        for feature in fulltext_features:
            gazetteer_file = os.path.join(gazetteers_dir, feature+".csv")
            count_file = os.path.join(feature_dir, "{0}_{1}.csv".format(dataset, feature))
            fulltext_features[feature](pmids, gazetteer_file, count_file, fulltext_dir)
            print("Completed {0} {1}".format(dataset, feature))
        
        # nbow and references
#        gazetteer_file = os.path.join(gazetteers_dir, "nbow.csv")
#        count_file = os.path.join(feature_dir, "{0}_nbow.csv".format(dataset))
#        extract_nbow(pmids, gazetteer_file, count_file, stemtext_dir)
#        print("Completed nbow")
        
#        gazetteer_file = os.path.join(gazetteers_dir, "references.csv")
#        count_file = os.path.join(feature_dir, "{0}_references.csv".format(dataset))
#        extract_cogat(pmids, gazetteer_file, count_file, reftext_dir)
#        print("Completed cogat")
    
    # Now a special step for CogAt weighting
    weighting_scheme = "ws2"
    directions = ["up", "down", "side"]
    weight_dfs = [[] for _ in directions]
    for i, dir_ in enumerate(directions):
        weight_dfs[i] = pd.read_csv(os.path.join(gazetteers_dir, "cogat_weights_{0}_{1}.csv".format(weighting_scheme, dir_)))
        weight_dfs[i].set_index("term", inplace=True)
        
    for dataset in datasets:
        os.rename(os.path.join(feature_dir, "{0}_cogat.csv".format(dataset)),
                  os.path.join(feature_dir, "{0}_cogat_count.csv".format(dataset)))
        count_file = os.path.join(feature_dir, "{0}_cogat_count.csv".format(dataset))
        count_df = pd.read_csv(count_file, index_col="pmid")
        weighted_file = os.path.join(feature_dir, "{0}_cogat.csv".format(dataset))
        apply_weights_recursively(count_df, weight_dfs, weighted_file)

if __name__ == "__main__":
    extract_features(sys.argv[1])

