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
from sklearn.cross_validation import KFold
import evaluate_classifier as ec
import copy
import shutil


def combine_features(feature_names, folder):
    """
    Produce combined count files for all possible combinations of features.
    """
    out_files = []
    path = os.path.join(folder, "train_")
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


def run_clf(feature_name, labels_dir, features_dir, out_folder):
    """
    Run sklearn OneVsRest multilabel classifier wrapped around a LinearSVC
    binary classifier with l2 penalty and 1.0 C on a given feature count file.
    
    Cross-validation is used to evaluate results.
    """
    # Train
    features_file = os.path.join(features_dir, "train_"+feature_name+".csv")
    labels_file = os.path.join(labels_dir, "train.csv")
    
    features_df = pd.read_csv(features_file)
    features = features_df.as_matrix()[:, 1:]
    labels_df = pd.read_csv(labels_file)
    labels = labels_df.as_matrix()[:, 1:]
    
    kf = KFold(labels.shape[0], 10)
    
    k = 0
    for train_index, test_index in kf:
        features_train, features_test = features[train_index, :], features[test_index, :]
        labels_train, labels_test = labels[train_index, :], labels[test_index, :]
        labels_test_df = labels_df.iloc[test_index]
        classif = OneVsRestClassifier(LinearSVC(penalty="l2", C=1.0))
        classif.fit(features_train, labels_train)
    
        # Test
        predictions = classif.predict(features_test)
        out_file = os.path.join(out_folder, "{}_k{}.csv".format(feature_name, k))
        np.savetxt(out_file, predictions, delimiter=",")
        
        # Evaluate
        metrics = ec.return_metrics(labels_test, predictions)
        lb_df = ec.return_labelwise(labels_test_df, predictions)
        lb_df.set_index("Label", inplace=True)
        
        if k == 0:
            average_array = np.zeros((len(kf), len(metrics)))
            lb_df_average = copy.deepcopy(lb_df)
        else:
            lb_df_average += lb_df
        average_array[k, :] = metrics
        k += 1
    metrics_average = list(np.mean(average_array, axis=0))
    lb_df_average /= len(kf)
    return metrics_average, lb_df_average


def run_feature_selection():
    """
    Create all possible combination feature count files, run simple classifier
    on each, and output summary statistics.
    """
    data_dir = "/Users/salo/NBCLab/athena-data/"
    
    labels_dir = os.path.join(data_dir, "labels/")
    features_dir = os.path.join(data_dir, "features/")
    fs_dir = os.path.join(data_dir, "feature_selection/")
    
    fs_predictions_dir = os.path.join(fs_dir, "predictions/")
    fs_features_dir = os.path.join(fs_dir, "features/")
    fs_labels_dir = os.path.join(fs_dir, "labels/")
    fs_results_dir = os.path.join(fs_dir, "results/")
    
#    features = ["ay", "cogat", "j", "kw", "nbow", "ref", "tw"]  # Full feature list
    features = ["authoryear", "journal", "titlewords"]  # Easily generated features to test functions
    
    # Copy files to feature selection directory
    source_file = os.path.join(labels_dir, "train.csv")
    dest_file = os.path.join(fs_labels_dir, "train.csv")
    shutil.copyfile(source_file, dest_file)
    for feature in features:
        source_file = os.path.join(features_dir, "train_{0}.csv".format(feature))
        dest_file = os.path.join(fs_features_dir, "train_{0}.csv".format(feature))
        shutil.copyfile(source_file, dest_file)
    
    # Combine features and run classifier
    combos = []
    for i in range(1, len(features)):
        combos += sorted(list(itertools.combinations(features, i)))
    
    combos = [list(tup) for tup in combos]
    combos += [features]
    
    out_metrics = []
    for combo in combos:
        feature_name = "_".join(combo)
        combine_features(combo, fs_features_dir)
        metrics, lb_df = run_clf(feature_name, fs_labels_dir, fs_features_dir,
                                 fs_predictions_dir)
        metrics.insert(0, feature_name)
        out_metrics += [metrics]
        lb_df.to_csv(os.path.join(fs_results_dir, "{0}_labelwise.csv".format(feature_name)))
    out_df = pd.DataFrame(columns=["Model", "F1 (macro-averaged by example)",
                                   "Macro Precision", "Micro Precision",
                                   "Macro Recall", "Micro Recall",
                                   "Hamming Loss"], data=out_metrics)
    out_df.to_csv(os.path.join(fs_results_dir, "results.csv"), index=False)

