# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:25:01 2016
Glossary:
- Features:
  - ay: Authors/Year
  - j: Journal
  - cogat: Cognitive Atlas
  - ref: References
  - nbow: Naive bag of words
  - kw: PubMed keywords
  - tw: Title words

@author: salo
"""

import numpy as np
import itertools
import pandas as pd
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import evaluate_classifier as ec


def combine_features(feature_names, folder):
    """
    Produce combined count files for all possible combinations of features.
    """
    out_files = []
    for dataset in ["train", "test"]:
        path = os.path.join(folder, dataset+"_features_")
        feature_files = [path+feature_name+".csv" for feature_name in feature_names]
        out_name = path + "_".join(feature_names) + ".csv"
        if out_name != feature_files[0]:
            feature_dfs = [[] for i in feature_names]
            for i, feature_file in enumerate(feature_files):
                feature_dfs[i] = pd.read_csv(feature_file, dtype=float)
                feature_dfs[i] = feature_dfs[i].set_index("pmid")
            feature_df = pd.concat(feature_dfs, axis=1, ignore_index=False)
            feature_df.to_csv(out_name)
        out_files += [out_name]


def run_clf(feature_name, in_folder, out_folder):
    """
    Run sklearn OneVsRest multilabel classifier wrapped around a LinearSVC
    binary classifier with l2 penalty and 1.0 C on a given feature count file.
    """
    # Train
    train_features_file = os.path.join(in_folder, "train_features_"+feature_name+".csv")
    train_labels_file = os.path.join(in_folder, "train_labels.csv")
    test_features_file = os.path.join(in_folder, "train_features_"+feature_name+".csv")
    test_labels_file = os.path.join(in_folder, "train_labels.csv")
    
    train_features_df = pd.read_csv(train_features_file)
    train_features = train_features_df.as_matrix()[:, 1:]
    train_labels_df = pd.read_csv(train_labels_file)
    train_labels = train_labels_df.as_matrix()[:, 1:]
    
    test_features_df = pd.read_csv(test_features_file)
    test_features = test_features_df.as_matrix()[:, 1:]
    
    classif = OneVsRestClassifier(LinearSVC(penalty="l2", C=1.0))
    classif.fit(train_features, train_labels)
    
    # Test
    test_pred = classif.predict(test_features)
    out_file = os.path.join(out_folder, feature_name+".csv")
    np.savetxt(out_file, test_pred, delimiter=",")
    
    # Evaluate
    metrics = ec.return_metrics(test_labels_file, test_pred)
    lb_df = ec.return_labelwise(test_labels_file, test_pred)
    return metrics, lb_df


def run_feature_selection():
    """
    Create all possible combination feature count files, run simple classifier
    on each, and output summary statistics.
    """
    in_dir = "/Users/salo/NBCLab/athena-data/processed_data/"
    out_dir = "/Users/salo/NBCLab/athena-data/feature_selection/"
#    features = ["ay", "cogat", "j", "kw", "nbow", "ref", "tw"]  # Full feature list
    features = ["ay", "j", "tw"]  # Easily generated features to test functions
    combos = []
    for i in range(1, len(features)):
        combos += sorted(list(itertools.combinations(features, i)))
    
    combos = [list(tup) for tup in combos]
    combos += [features]
    
    out_metrics = []
    for combo in combos:
        feature_name = "_".join(combo)
        combine_features(combo, in_dir)
        metrics, lb_df = run_clf(feature_name, in_dir, out_dir)
        metrics.insert(0, feature_name)
        out_metrics += [metrics]
    out_df = pd.DataFrame(columns=["Model", "F1 (macro-averaged by example)",
                                   "Macro Precision", "Micro Precision",
                                   "Macro Recall", "Micro Recall",
                                   "Hamming Loss"], data=out_metrics)
    
    out_df.to_csv(os.path.join(out_dir, "fs_results.csv"), index=False)
    lb_df.to_csv(os.path.join(out_dir, "labelwise.csv"), index=False)

