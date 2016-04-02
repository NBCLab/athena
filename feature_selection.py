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
- Sources:
  - c: Combined abstract and methods section
  - m: Methods section
  - a: Abstract
  - f: Full text

@author: salo
"""

import itertools
import pandas as pd
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import evaluate_classifier as ec


def combine_features(feature_names, folder):
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


def run_clf(feature_name, folder):
    # Train
    train_features_file = os.path.join(folder, "train_features_"+feature_name+".csv")
    train_labels_file = os.path.join(folder, "train_labels.csv")
    test_features_file = os.path.join(folder, "train_features_"+feature_name+".csv")
    test_labels_file = os.path.join(folder, "train_labels.csv")
    
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
    
    # Evaluate
    metrics = ec.return_metrics(test_labels_file, test_pred)
    return metrics


def run_feature_selection():
    dir_ = "/Users/salo/NBCLab/athena-data/processed_data/"
    features = ["ay", "cogat", "j", "kw", "nbow", "ref", "tw"]
    combos = []
    for i in range(1, len(features)):
        combos += sorted(list(itertools.combinations(features, i)))
    
    combos = [list(tup) for tup in combos]
    combos += [features]
    
    out_metrics = []
    for combo in combos:
        feature_name = "_".join(combo)
        combine_features(combo, dir_)
        metrics = run_clf(feature_name, dir_)
        metrics.insert(0, feature_name)
        out_metrics += [metrics]
    out_df = pd.DataFrame(columns=["Model", "F1 (macro-averaged by example)",
                                   "Macro Precision", "Micro Precision",
                                   "Macro Recall", "Micro Recall",
                                   "Hamming Loss"], data=out_metrics)
    
    out_df.to_csv(os.path.join(dir_, "fs_results.csv"), index=False)
