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


def return_metrics(train_label_file, predictions_file):
    """
    Calculate metrics for model based on predicted labels.
    """
    df = pd.read_csv(train_label_file, dtype=int)
    true_labels = df.as_matrix()[:, 1:]
    
    predictions = np.loadtxt(predictions_file, dtype=int, delimiter=",")
    
    macro_precision = precision_score(true_labels, predictions, average="macro")
    micro_precision = precision_score(true_labels, predictions, average="micro")
    
    macro_recall = recall_score(true_labels, predictions, average="macro")
    micro_recall = recall_score(true_labels, predictions, average="micro")
    
    hamming_loss_ = hamming_loss(true_labels, predictions)
    
    macro_f1_score_by_example = f1_score(true_labels, predictions, average="samples")
    metrics = [macro_f1_score_by_example, macro_precision, micro_precision, macro_recall, micro_recall, hamming_loss_]
    return metrics


def return_all(train_label_file, predictions_dir):
    """
    Calculate the metrics for all csv files in the folder, except for
    compiled.csv.
    """
    out_metrics = []
    
    predictions_files = glob(os.path.join(predictions_dir, "*.csv"))
    predictions_files = [file_ for file_ in predictions_files if not "compiled" in file_]
    for predictions_file in predictions_files:
        model_name, _ = os.path.splitext(predictions_file)
        model_name = os.path.basename(model_name)
        metrics = return_metrics(train_label_file, predictions_file)
        metrics.insert(0, model_name)
        out_metrics += [metrics]
    out_df = pd.DataFrame(columns=["Model", "F1 (macro-averaged by example)",
                                   "Macro Precision", "Micro Precision",
                                   "Macro Recall", "Micro Recall",
                                   "Hamming Loss"], data=out_metrics)
    return out_df


def test():
    train_label_file = "/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv"
    predictions_file = "/Users/salo/NBCLab/athena-data/predictions/predictions.csv"
    predictions_dir = "/Users/salo/NBCLab/athena-data/predictions/"

    metrics = return_metrics(train_label_file, predictions_file)
    f1, mac_prec, mic_prec, mac_rec, mic_rec, hl = metrics
    out_df = return_all(train_label_file, predictions_dir)
    out_df.to_csv("/Users/salo/NBCLab/athena-data/predictions/compiled.csv", index=False)
