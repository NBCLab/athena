# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:40:23 2016

@author: salo
"""

import os
import pandas as pd
import numpy as np


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

folder = "/home/tsalo006/cogpo/athena-data/meta_data/"
filenames = ["ExecutiveFunction.csv", "Face.csv", "Pain.csv", "Passive.csv",
             "Reward.csv", "Semantic.csv", "Word.csv", "nBack.csv"]
data_dir = "/home/tsalo006/cogpo/athena-data/combined/"
data_dir2 = "/home/tsalo006/cogpo/athena-data/testData/executiveData/"

#folder = "/Users/salo/NBCLab/athena-data/meta_data/"
#data_dir = "/Users/salo/NBCLab/athena-data/combined/"
#data_dir2 = "/Users/salo/NBCLab/athena-data/testData/executiveData/"

column_to_cogpo = {"Paradigm Class": "Experiments.ParadigmClass",
                   "Behavioral Domain": "Experiments.BehavioralDomain",}
#                   "Diagnosis": "Subjects.Diagnosis",
#                   "Stimulus Modality": "Conditions.StimulusModality",
#                   "Stimulus Type": "Conditions.StimulusType",
#                   "Response Modality": "Conditions.OvertResponseModality",
#                   "Response Type": "Conditions.OvertResponseType",
#                   "Instructions": "Conditions.Instruction"}

full_cogpo = []
dfs = [pd.read_csv(os.path.join(folder, file_), dtype=str)[["PubMed ID"] + column_to_cogpo.keys()] for file_ in filenames]
df = pd.concat(dfs, ignore_index=True)

for column in column_to_cogpo.keys():
    table = df_to_list(df, column, column_to_cogpo[column])
    full_cogpo += table

full_cogpo = sorted(list(set(full_cogpo)))

# Preallocate label DataFrame
df = df[df["PubMed ID"].str.contains("^\d+$")].reset_index()
list_of_pmids = df["PubMed ID"].unique().tolist()
list_of_files = os.listdir(data_dir) + os.listdir(data_dir2)
list_of_files = [os.path.splitext(file_)[0] for file_ in list_of_files]
list_of_files = sorted(list(set(list_of_files)))
list_of_pmids = sorted(list(set(list_of_pmids).intersection(list_of_files)))

column_names = ["pmid"] + full_cogpo
df2 = pd.DataFrame(columns=column_names,
                   data=np.zeros((len(list_of_pmids), len(column_names))))
df2["pmid"] = list_of_pmids

for row in df.index:
    pmid = df["PubMed ID"].iloc[row]
    if pmid in list_of_pmids:
        for column in column_to_cogpo.keys():
            values = df[column].iloc[row]
            if pd.notnull(values):
                values = values.split("| ")
                values = ["{0}.{1}".format(column_to_cogpo[column], clean_str(item)) for item in values]
                for value in values:
                    for out_column in df2.columns:
                        if out_column in value:
                            ind = df2.loc[df2["pmid"]==pmid].index[0]
                            df2[out_column].iloc[ind] = 1

# Reduce DataFrame
label_counts = df2.sum()
rep_labels = label_counts[label_counts>4].index
df2 = df2[rep_labels]
df2 = df2[(df2.T != 0).any()]

df2.to_csv("/home/tsalo006/cogpo/all_labels.csv", index=False)
