# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:10:41 2016

@author: tsalo006
"""
import os
from glob import glob
import pandas as pd
import random
import numpy as np
from utils import clean_str, df_to_list

data_dir = "/home/data/nbc/athena/athena-data/"
metadata_dir = os.path.join(data_dir, "metadata/")
filenames = sorted(glob(os.path.join(metadata_dir, "*.csv")))
text_dir = os.path.join(data_dir, "text/full/")

column_to_cogpo = {"Paradigm Class": "Experiments.ParadigmClass",
                   "Behavioral Domain": "Experiments.BehavioralDomain",
                   "Diagnosis": "Subjects.Diagnosis",
                   "Stimulus Modality": "Conditions.StimulusModality",
                   "Stimulus Type": "Conditions.StimulusType",
                   "Response Modality": "Conditions.OvertResponseModality",
                   "Response Type": "Conditions.OvertResponseType",
                   "Instructions": "Conditions.Instruction",
                   "Context": "Experiments.Context"}

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
label_df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
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
label_df = label_df[(label_df.T != 0).any()]
label_df = label_df.astype(int).astype(str)
print label_df.shape
out_file = os.path.join(data_dir, "labels/full_fullcogpo.csv")
label_df.to_csv(out_file, index=False)