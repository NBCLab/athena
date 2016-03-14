# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:40:23 2016

@author: salo
"""

import os
import pandas as pd
import numpy as np


def df_to_list(df, column_name, prefix):
    table = df[pd.notnull(df[column_name])][column_name]
    table.apply(lambda x: "{%s}" % "| ".join(x))
    table = table.tolist()
    table = [item.replace(" ", "").replace("'", "") for sublist in
             table for item in sublist.split("| ")]
    parents = table
    while parents:
        parents = [".".join(item.split(".")[:-1]) for item in parents if len(item.split("."))>1]
        table += parents
    table = ["{0}.{1}".format(prefix, item) for item in table]
    return table

train_labels = "/home/tsalo006/cogpo/train_labels.csv"
folder = "/home/tsalo006/cogpo/"
filenames = ["KarinaAnnotationsDataComplete.csv"]
             
column_to_cogpo = {"Paradigm Class": "Experiments.ParadigmClass",
                   "Behavioral Domain": "Experiments.BehavioralDomain",
                   "Diagnosis": "Subjects.Diagnosis",
                   "Stimulus Modality": "Conditions.StimulusModality",
                   "Stimulus Type": "Conditions.StimulusType",
                   "Response Modality": "Conditions.OvertResponseModality",
                   "Response Type": "Conditions.OvertResponseType",
                   "Instructions": "Conditions.Instruction"}

file_ = filenames[0]
for i, file_ in enumerate(filenames):
    full_file = os.path.join(folder, file_)
    df_temp = pd.read_csv(full_file, dtype=str)
    if i == 0:
        df = df_temp
    else:
        df = pd.concat([df, df_temp], ignore_index=True)

train_df = pd.read_csv(train_labels)
full_cogpo = train_df.columns.values[1:]

# Preallocate label DataFrame
df = df[df["PubMed ID"].str.contains("^\d+$")].reset_index()
list_of_pmids = df["PubMed ID"].unique().tolist()

column_names = ["pmid"] + full_cogpo
df2 = pd.DataFrame(columns=column_names,
                   data=np.zeros((len(list_of_pmids), len(column_names))))
df2["pmid"] = list_of_pmids

for row in df.index:
    pmid = df["PubMed ID"].iloc[row]
    for column in column_to_cogpo.keys():
        values = df[column].iloc[row]
        if pd.notnull(values):
            values = values.split("| ")
            values = ["{0}.{1}".format(column_to_cogpo[column], item.replace(" ", "").replace("'", "")) for item in values]
            for value in values:
                for out_column in df2.columns:
                    if out_column in value:
                        ind = df2.loc[df2["pmid"]==pmid].index[0]
                        df2[out_column].iloc[ind] = 1

df2.to_csv("/home/tsalo006/cogpo/test_labels.csv", index=False)
