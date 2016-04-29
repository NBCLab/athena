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
import os


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

# Primary labels
primary_labels = ["pmid",
                  "Experiments.ParadigmClass.FaceMonitor/Discrimination",
                  "Experiments.ParadigmClass.Reward",
                  "Experiments.ParadigmClass.SemanticMonitor/Discrimination",
                  "Experiments.ParadigmClass.WordGeneration",
                  "Experiments.ParadigmClass.n-back"]

# Run function for both datasets
data_dir = "/home/data/nbc/athena/v1.1-data/"
train_labels = os.path.join(data_dir, "labels/train.csv")
train_features = os.path.join(data_dir, "features/train_cogat.csv")
test_labels = os.path.join(data_dir, "labels/test.csv")
test_features = os.path.join(data_dir, "features/test_cogat.csv")

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
out_df.to_csv(os.path.join(data_dir, "statistics/dataset_statistics.csv"))

# Also output file with labels
labels = train_label_df.columns.tolist()[1:]
out_df = pd.DataFrame(columns=["Label"], data=labels)
out_df.to_csv(os.path.join(data_dir, "labels/labels.csv"), index=False)

# Limit to primary labels and output reduced file
train_label_df = train_label_df[primary_labels]
test_label_df = test_label_df[primary_labels]

print("Primary training dataset statistics:")
train_df = statistics(train_label_df, train_feature_df, "Training")

print("Primary test dataset statistics:")
test_df = statistics(test_label_df, test_feature_df, "Test")

out_df = pd.concat([train_df, test_df])
out_df.to_csv(os.path.join(data_dir, "statistics/dataset_statistics_primary.csv"))
