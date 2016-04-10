# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:40:23 2016

@author: salo
"""

import os
import pandas as pd
import numpy as np
from utils import cogpo_columns
from glob import glob


def clean_str(str_):
    label = str_.replace(" ", "").replace("'", "").replace("(", ".").replace(")", "").replace("Stroop-", "Stroop.")
    return label


def df_to_list(df, column_name, prefix):
    table = df[pd.notnull(df[column_name])][column_name]
    table.apply(lambda x: "{%s}" % "| ".join(x))
    table = table.tolist()
    table = [clean_str(item) for sublist in table for item in sublist.split("| ")]
    
    parents = table
    while parents:
        parents = [".".join(item.split(".")[:-1]) for item in parents if len(item.split("."))>1]
        table += parents
    table = ["{0}.{1}".format(prefix, item) for item in table]
    return table


def label_data(data_dir="/Users/salo/NBCLab/athena-data/"):
    """
    Convert metadata files to instance-by-label matrix.
    """
    metadata_dir = os.path.join(data_dir, "metadata/")
    filenames = sorted(glob(os.path.join(metadata_dir, "*.csv")))
    text_dir = os.path.join(data_dir, "/text/full/")
    
    columns = ["Paradigm Class", "Behavioral Domain"]
    column_to_cogpo = cogpo_columns(columns)
    
    full_cogpo = []
    metadata_dfs = [pd.read_csv(file_, dtype=str)[["PubMed ID"] + column_to_cogpo.keys()] for file_ in filenames]
    metadata_df = pd.concat(metadata_dfs, ignore_index=True)
    
    for column in column_to_cogpo.keys():
        table = df_to_list(metadata_df, column, column_to_cogpo[column])
        full_cogpo += table
    
    full_cogpo = sorted(list(set(full_cogpo)))
    
    # Preallocate label DataFrame
    metadata_df = metadata_df[metadata_df["PubMed ID"].str.contains("^\d+$")].reset_index()
    list_of_pmids = metadata_df["PubMed ID"].unique().tolist()
    list_of_files = [os.path.splitext(file_)[0] for file_ in os.listdir(text_dir)]
    list_of_files = sorted(list(set(list_of_files)))
    list_of_pmids = sorted(list(set(list_of_pmids).intersection(list_of_files)))
    
    column_names = ["pmid"] + full_cogpo
    label_df = pd.DataFrame(columns=column_names,
                            data=np.zeros((len(list_of_pmids), len(column_names))))
    label_df["pmid"] = list_of_pmids
    
    for row in metadata_df.index:
        pmid = metadata_df["PubMed ID"].iloc[row]
        if pmid in list_of_pmids:
            for column in column_to_cogpo.keys():
                values = metadata_df[column].iloc[row]
                if pd.notnull(values):
                    values = values.split("| ")
                    values = ["{0}.{1}".format(column_to_cogpo[column], clean_str(item)) for item in values]
                    for value in values:
                        for out_column in label_df.columns:
                            if out_column in value:
                                ind = label_df.loc[label_df["pmid"]==pmid].index[0]
                                label_df[out_column].iloc[ind] = 1
    
    # Reduce DataFrame
    label_counts = label_df.sum()
    keep_labels = label_counts[label_counts>4].index
    label_df = label_df[keep_labels]
    label_df = label_df[(label_df.T != 0).any()]
    
    out_file = os.path.join(data_dir, "labels/full.csv")
    label_df.to_csv(out_file, index=False)
