# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 00:07:14 2016

Prepare data for extraction and analysis.

Inputs:
- IDs
- Raw text
- Raw labels (metadata)

Outputs:
- Processed text
- Labeled data
- Gazetteers

@author: salo
"""

import os
import pandas as pd
import gazetteers
import process_data


def process_raw_data(data_dir="/home/data/nbc/athena/v1.1-data/"):
    """
    Label data using metadata files, split data into training and test
    datasets.
    """
    process_data.stem_corpus(data_dir)
    process_data.label_data(data_dir)
    for text_type in ["full", "combined"]:
        type_dir = os.path.join(data_dir, text_type)
        labels_file = os.path.join(type_dir, "labels/full.csv")
        process_data.split_data(labels_file, test_percent=0.33)


def generate_gazetteers(data_dir="/home/data/nbc/athena/v1.1-data/"):
    """
    Creates list of unique terms for four gazetteers derived from metadata
    available through PubMed:
        - Authors and year of publication
        - Journal of publication
        - Words in title (not including stopwords)
        - Author-generated keywords (if available on PubMed)
            - This includes multiword expressions.
    """
    text_dir = os.path.join(data_dir, "text/")
    for text_type in ["full", "combined"]:
        type_dir = os.path.join(data_dir, text_type)
        label_file = os.path.join(type_dir, "labels/full.csv")
        gaz_dir = os.path.join(type_dir, "gazetteers/")
        stem_text_dir = os.path.join(text_dir, "stemmed_"+)

        df = pd.read_csv(label_file)
        pmids = df["pmid"].astype(str).tolist()
    
        nbow_gaz = gazetteers.generate_nbow_gazetteer(pmids, stem_text_dir)
        print("Completed nbow gaz.")

        # Save gazetteer
        gazetteers.save_gaz(nbow_gaz, gaz_dir, "nbow")
