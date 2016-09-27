# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:10:19 2016

Functions to evaluate multilabel classifier predictions.

Measures to use:
- F1 score, macro-averaged
- Precision, macro- and micro-averaged
- Recall, macro- and micro-averaged
- Hamming Loss, macro- and micro-averaged

@author: salo
"""

import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss


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


def dataset_statistics(data_dir="/home/data/nbc/athena/athena-data/", feature_name="authoryear"):
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
    
    macro_f1 = f1_score(labels, predictions, average="macro")
    micro_f1 = f1_score(labels, predictions, average="micro")
    
    macro_precision = precision_score(labels, predictions, average="macro")
    micro_precision = precision_score(labels, predictions, average="micro")
    
    macro_recall = recall_score(labels, predictions, average="macro")
    micro_recall = recall_score(labels, predictions, average="micro")
    
    hamming_loss_ = hamming_loss(labels, predictions)
    
    metrics = [macro_f1, micro_f1, macro_precision, micro_precision,
               macro_recall, micro_recall, hamming_loss_]
    return metrics


def return_primary(labels, predictions, label_names=None):
    """
    Calculate metrics for model based on predicted labels. But only for
    primary labels.
    """
    # Primary labels
    primary_labels = ["ParadigmClass.FaceMonitor/Discrimination",
                      "ParadigmClass.Reward",
                      "ParadigmClass.SemanticMonitor/Discrimination",
                      "ParadigmClass.WordGeneration",
                      "ParadigmClass.n-back",
                      "ParadigmClass.PainMonitor/Discrimination"]
    
    if isinstance(labels, str):
        df = pd.read_csv(labels, dtype=int)
        col_idx = np.where(df.columns.isin(primary_labels))[0]
        col_idx -= 1
        labels = df.as_matrix()[:, 1:]
    else:
        col_idx = np.where(label_names.isin(primary_labels))[0]
        col_idx -= 1
    labels = labels[:, col_idx]
    predictions = predictions[:, col_idx]
    
    macro_f1 = f1_score(labels, predictions, average="macro")
    micro_f1 = f1_score(labels, predictions, average="micro")
    
    macro_precision = precision_score(labels, predictions, average="macro")
    micro_precision = precision_score(labels, predictions, average="micro")
    
    macro_recall = recall_score(labels, predictions, average="macro")
    micro_recall = recall_score(labels, predictions, average="micro")
    
    hamming_loss_ = hamming_loss(labels, predictions)

    metrics = [macro_f1, micro_f1, macro_precision, micro_precision,
               macro_recall, micro_recall, hamming_loss_]
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
        print "Labels is unrecognized type {0}".format(type(labels))
        raise Exception()
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


def return_all(labels_file, predictions_dir):
    """
    Calculate the metrics for all csv files in the folder, except for
    compiled.csv.
    """
    out_metrics = []
    
    predictions_files = glob(os.path.join(predictions_dir, "*.csv"))
    for predictions_file in predictions_files:
        model_name, _ = os.path.splitext(predictions_file)
        model_name = os.path.basename(model_name)
        
        predictions = np.loadtxt(predictions_file, dtype=int, delimiter=",")
        
        metrics = return_metrics(labels_file, predictions)
        metrics.insert(0, model_name)
        out_metrics += [metrics]
    out_df = pd.DataFrame(columns=["Model", "Macro F1", "Micro F1",
                                   "Macro Precision", "Micro Precision",
                                   "Macro Recall", "Micro Recall",
                                   "Hamming Loss"], data=out_metrics)
    return out_df


def test():
    train_label_file = "/home/data/nbc/athena/athena-data/labels/train.csv"
    predictions_file = "/home/data/nbc/athena/athena-data/predictions/predictions.csv"
    predictions_dir = "/home/data/nbc/athena/athena-data/predictions/"
    predictions = np.loadtxt(predictions_file, dtype=int, delimiter=",")
    
    metrics = return_metrics(train_label_file, predictions)
    f1, mac_prec, mic_prec, mac_rec, mic_rec, hl = metrics
    out_df = return_all(train_label_file, predictions_dir)
    out_df.to_csv("/home/data/nbc/athena/athena-data/statistics/metrics.csv", index=False)
    labelwise_df = return_labelwise(train_label_file, predictions_dir)
    labelwise_df.to_csv("/home/data/nbc/athena/athena-data/statistics/labelwise_metrics.csv",
                        index=False)

    df = pd.read_csv(train_label_file, dtype=int)
    


