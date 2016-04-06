# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:43:09 2016
Provides summary statistics common to multilabel datasets for both the
training and test datasets:
    - Number of instances
    - Number of features (same across datasets)
    - Number of labels (same across datasets)
    - Label cardinality
    - Label density
    - Number of unique labelsets

@author: salo
"""
from __future__ import division
import pandas as pd
import numpy as np


def statistics(label_df, feature_df, dataset_name):
    out_df = pd.DataFrame(columns=["Number of Instances",
                                   "Number of Features", "Number of Labels",
                                   "Label Cardinality", "Label Density",
                                   "Number of Unique Labelsets"],
                          index=[dataset_name])
    out_df.index.name = "Dataset"
    
    feature_df = feature_df.drop("pmid", axis=1)
    features = feature_df.columns.tolist()
    
    label_df = label_df.drop("pmid", axis=1)
    labels = label_df.columns.tolist()
    
    n_instances = len(label_df)
    print("\tNumber of instances: {0}".format(n_instances))
    
    n_features = len(features)
    print("\tNumber of features: {0}".format(n_features))
    
    n_labels = len(labels)
    print("\tNumber of labels: {0}".format(n_labels))
    
    label_cardinality = label_df.sum(axis=1).sum() / n_instances
    print("\tLabel cardinality: {0}".format(label_cardinality))
    
    n_positive = label_df.sum(axis=0).sum()
    label_density = (label_df.sum(axis=1) / n_positive).sum() / len(labels)
    print("\tLabel density: {0}".format(label_density))
    
    label_array = label_df.values
    unique_labelsets = np.vstack({tuple(row) for row in label_array})
    n_unique_labelsets = unique_labelsets.shape[0]
    print("\tNumber of unique labelsets: {0}".format(n_unique_labelsets))
    
    row = [n_instances, n_features, n_labels, label_cardinality, label_density, n_unique_labelsets]
    out_df.loc[dataset_name] = row
    return out_df

# Run function for both datasets
train_labels = "/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv"
train_features = "/Users/salo/NBCLab/athena-data/processed_data/train_features_ay.csv"
test_labels = "/Users/salo/NBCLab/athena-data/processed_data/test_labels.csv"
test_features = "/Users/salo/NBCLab/athena-data/processed_data/test_features_ay.csv"

# Load and combine data
train_label_df = pd.read_csv(train_labels, dtype=int)
train_feature_df = pd.read_csv(train_features, dtype=float)

test_label_df = pd.read_csv(test_labels, dtype=int)
test_feature_df = pd.read_csv(test_features, dtype=float)

print("Training dataset statistics:")
train_df = statistics(train_label_df, train_feature_df, "Training")

print("Test dataset statistics:")
test_df = statistics(test_label_df, test_feature_df, "Test")

out_df = pd.concat([train_df, test_df])
out_df.to_csv("/Users/salo/NBCLab/athena-data/dataset_statistics.csv")
