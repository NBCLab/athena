# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:46:36 2016

Convert pandas DataFrames to arff format.

@author: salo
"""

import pandas as pd

train_features = "/Users/salo/NBCLab/athena-data/processed_data/train_features_ay.csv"
train_labels = "/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv"

# Load and combine data
feature_df = pd.read_csv(train_features, dtype=float)
feature_df = feature_df.set_index("pmid")
feature_df.index.names = [None]
features = feature_df.columns.tolist()

label_df = pd.read_csv(train_labels, dtype=int)
label_df = label_df.set_index("pmid")
label_df.index.names = [None]
labels = label_df.columns.tolist()

out_string = "@relation TrainingData\n"

for feature in features:
    if " " in feature:
        out_string += '@attribute "{0}" numeric\n'.format(feature)
    else:
        out_string += '@attribute {0} numeric\n'.format(feature)

for label in labels:
    if " " in label:
        out_string += '@attribute "{0}" {{0, 1}}\n'.format(label)
    else:
        out_string += '@attribute {0} {{0, 1}}\n'.format(label)

out_string += "\n@data\n"

for pmid in label_df.index.values:
    feature_list = feature_df.loc[pmid].tolist()
    feature_str = ",".join(map(str, feature_list))
    
    label_list = label_df.loc[pmid].tolist()
    label_str = ",".join(map(str, label_list))
    
    out_string += "{0},{1}\n".format(feature_str, label_str)

with open("/Users/salo/NBCLab/athena-data/processed_data/train_data.arff", "w") as fo:
    fo.write(out_string)