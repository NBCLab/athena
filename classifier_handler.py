# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 00:07:14 2016

Manages MEKA and MULAN classifiers for ATHENA scientific article
classification.

Inputs:
- Count files
- Labels

Outputs:
- Models
- Predictions
- Statistics

@author: salo
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss


def combine_features(feature_names, data_dir="/home/data/nbc/athena/v1.1-data/"):
    """
    Produce combined count files for selected features.
    """
    features_dir = os.path.join(data_dir, "features/")
    datasets = ["train", "test"]
    for dataset in datasets:
        path = os.path.join(features_dir, "train_")
        out_name = path + "_".join(feature_names) + ".csv"
        feature_files = [path+fn+".csv" for fn in feature_names]
        feature_dfs = [pd.read_csv(ff, dtype=float, index_col="pmid") for ff in feature_files]
        feature_df = pd.concat(feature_dfs, axis=1, ignore_index=False)
        feature_df.to_csv(out_name)


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
    n_features = len(features)    
    n_labels = len(labels)    
    label_cardinality = label_df.sum(axis=1).sum() / n_instances    
    n_positive = label_df.sum(axis=0).sum()
    label_density = (label_df.sum(axis=1) / n_positive).sum() / len(labels)
    
    label_array = label_df.values
    unique_labelsets = np.vstack({tuple(row) for row in label_array})
    n_unique_labelsets = unique_labelsets.shape[0]
    
    row = [n_instances, n_features, n_labels, label_cardinality, label_density, n_unique_labelsets]
    out_df.loc[dataset_name] = row
    return out_df


def dataset_statistics(data_dir="/home/data/nbc/athena/v1.1-data/", feature_name="authoryear"):
    labels_dir = os.path.join(data_dir, "labels")
    features_dir = os.path.join(data_dir, "features")
    statistics_file = os.path.join(data_dir, "statistics/dataset_statistics.csv")
    
    # Run function for both datasets
    datasets = ["train", "test"]
    dfs = [[] for _ in datasets]
    for i, dataset in enumerate(datasets):
        labels_file = os.path.join(labels_dir, "{0}.csv".format(dataset))
        features_file = os.path.join(features_dir, "{0}_{1}.csv".format(dataset, feature_name))
        labels = pd.read_csv(labels_file, dtype=int)
        features = pd.read_csv(features_file, dtype=float)
        dfs[i] = statistics(labels, features, dataset)

    out_df = pd.concat(dfs)
    out_df.to_csv(statistics_file)


def return_metrics(labels, predictions):
    """
    Calculate metrics for model based on predicted labels.
    """
    if isinstance(labels, str):
        df = pd.read_csv(labels, dtype=int)
        labels = df.as_matrix()[:, 1:]
    
    macro_precision = precision_score(labels, predictions, average="macro")
    micro_precision = precision_score(labels, predictions, average="micro")
    
    macro_recall = recall_score(labels, predictions, average="macro")
    micro_recall = recall_score(labels, predictions, average="micro")
    
    hamming_loss_ = hamming_loss(labels, predictions)
    
    macro_f1_score_by_example = f1_score(labels, predictions, average="samples")
    metrics = [macro_f1_score_by_example, macro_precision, micro_precision, macro_recall, micro_recall, hamming_loss_]
    return metrics


def return_labelwise(labels, predictions):
    """
    Calculate metrics for each label in model.
    """
    if isinstance(labels, str):
        df = pd.read_csv(labels, dtype=int)
    elif isinstance(labels, pd.DataFrame):
        df = labels
    else:
        raise Exception("Labels is unrecognized type {0}".format(type(labels)))
    
    label_names = list(df.columns.values)[1:]
    labels = df.as_matrix()[:, 1:]
    
    metrics = [f1_score, precision_score, recall_score, hamming_loss]
    metrics_array = np.zeros((len(label_names), len(metrics)))
    for i in range(len(label_names)):
        label_true = labels[:, i]
        label_pred = predictions[:, i]
        for j in range(len(metrics)):
            metrics_array[i, j] = metrics[j](label_true, label_pred)
    metric_df = pd.DataFrame(columns=["F1", "Precision", "Recall", "Hamming Loss"],
                            index=label_names, data=metrics_array)
    metric_df["Label"] = metric_df.index
    metric_df = metric_df[["Label", "F1", "Precision", "Recall", "Hamming Loss"]]
    return metric_df
