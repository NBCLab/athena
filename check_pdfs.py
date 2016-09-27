# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:19:43 2016

Determine intersection of metadata and pdfs.

@author: tsalo006
"""

import os
from glob import glob
import pandas as pd
import random
import numpy as np
from nltk.stem.porter import PorterStemmer
from utils import cogpo_columns, clean_str, df_to_list


data_dir="/home/data/nbc/athena/athena-data/"
metadata_dir = os.path.join(data_dir, "metadata/")
filenames = sorted(glob(os.path.join(metadata_dir, "*.csv")))
pdf_dir = os.path.join(data_dir, "pdfs/")

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
print len(list_of_pmids)
list_of_files = [os.path.splitext(file_)[0] for file_ in os.listdir(pdf_dir)]
list_of_files = sorted(list(set(list_of_files)))
print len(list_of_files)
list_of_pmids = sorted(list(set(list_of_pmids).intersection(list_of_files)))
print len(list_of_pmids)


