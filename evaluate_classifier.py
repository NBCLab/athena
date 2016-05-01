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
    primary_labels = ["Experiments.ParadigmClass.FaceMonitor/Discrimination",
                      "Experiments.ParadigmClass.Reward",
                      "Experiments.ParadigmClass.SemanticMonitor/Discrimination",
                      "Experiments.ParadigmClass.WordGeneration",
                      "Experiments.ParadigmClass.n-back"]
    
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
    data_dir = "/home/data/nbc/athena/v1.1-data/"
    
    for text_type in ["full", "combined"]:
        type_dir = os.path.join(data_dir, text_type)
        test_label_file = os.path.join(type_dir, "labels/test.csv")
        predictions_dir = os.path.join("predictions/")
        predictions = np.loadtxt(predictions_file, dtype=int, delimiter=",")
        
        metrics = return_metrics(test_label_file, predictions)
        f1, mac_prec, mic_prec, mac_rec, mic_rec, hl = metrics
        out_df = return_all(test_label_file, predictions_dir)
        out_df.to_csv(os.path.join(type_dir, "statistics/metrics.csv", index=False))
        labelwise_df = return_labelwise(test_label_file, predictions_dir)
        labelwise_df.to_csv(os.path.join(type_dir, "statistics/labelwise_metrics.csv", index=False))
    
        df = pd.read_csv(train_label_file, dtype=int)
