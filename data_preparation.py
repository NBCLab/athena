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
import cogat
import references
import process_data


def process_raw_data(data_dir="/home/data/nbc/athena/athena-data/"):
    """
    Label data using metadata files, split data into training and test
    datasets.
    """
    process_data.stem_corpus(data_dir)
    process_data.label_data(data_dir)
    labels_file = os.path.join(data_dir, "labels/full.csv")
    process_data.split_data(labels_file, test_percent=0.33)


def generate_gazetteers(data_dir="/home/data/nbc/athena/athena-data/"):
    """
    Creates list of unique terms for four gazetteers derived from metadata
    available through PubMed:
        - Authors and year of publication
        - Journal of publication
        - Words in title (not including stopwords)
        - Author-generated keywords (if available on PubMed)
            - This includes multiword expressions.
    """
    label_file = os.path.join(data_dir, "labels/full.csv")
    gaz_dir = os.path.join(data_dir, "gazetteers/")
    text_dir = os.path.join(data_dir, "text/")
    stem_text_dir = os.path.join(text_dir, "stemmed_full/")
    ref_text_dir = os.path.join(text_dir, "reference_data/")
    
    df = pd.read_csv(label_file)
    pmids = df["pmid"].astype(str).tolist()
    
    nbow_gaz = gazetteers.generate_nbow_gazetteer(pmids, stem_text_dir)
    print("Completed nbow gaz.")
    metadata_gazs = gazetteers.generate_metadata_gazetteers(pmids)
    authoryear_gaz, journal_gaz, keyword_gaz, titleword_gaz = metadata_gazs
    print("Completed metadata gaz.")
    references_df = references.generate_references_gazetteer(pmids, ref_text_dir)
    print("Completed references gaz.")
    
    cogat_df = cogat.create_id_sheet()
    rel_df = cogat.create_rel_sheet(cogat_df)
    
    weighting_schemes = ["ws2_up", "ws2_down", "ws2_side"]
    for weighting_scheme in weighting_schemes:
        weight_df = cogat.weight_rels(rel_df, weighting_scheme)
        weight_df.to_csv(os.path.join(gaz_dir, "cogat_weights_{0}.csv".format(weighting_scheme)),
                         index=True)
    print("Completed cogat gaz.")

    # Save gazetteers
    cogat_df.to_csv(os.path.join(gaz_dir, "cogat.csv"), index=False)
    rel_df.to_csv(os.path.join(gaz_dir, "cogat_relationships.csv"), index=False)
    references_df.to_csv(os.path.join(gaz_dir, "references.csv"), index=False)
    gazetteers.save_gaz(nbow_gaz, gaz_dir, "nbow")
    gazetteers.save_gaz(authoryear_gaz, gaz_dir, "authoryear")
    gazetteers.save_gaz(journal_gaz, gaz_dir, "journal")
    gazetteers.save_gaz(titleword_gaz, gaz_dir, "titlewords")
    gazetteers.save_gaz(keyword_gaz, gaz_dir, "keywords")
