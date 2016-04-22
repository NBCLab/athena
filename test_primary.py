# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:29:22 2016

@author: tsalo006
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
    label_names = labels_df.columns
    labels = labels_df.as_matrix()[:, 1:]
    
    kf = KFold(labels.shape[0], 10)
    
    for k, (train_index, test_index) in enumerate(kf):
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
        primary_metrics = ec.return_primary(labels_test, predictions, label_names)
        lb_df = ec.return_labelwise(labels_test_df, predictions)
        lb_df.set_index("Label", inplace=True)
        
        if k == 0:
            average_array = np.zeros((len(kf), len(metrics)))
            primary_array = np.zeros((len(kf), len(metrics)))
            lb_df_average = copy.deepcopy(lb_df)
        else:
            lb_df_average += lb_df
        average_array[k, :] = metrics
        primary_array[k, :] = primary_metrics
    metrics_average = list(np.mean(average_array, axis=0))
    primary_metrics_average = list(np.mean(primary_array, axis=0))
    print metrics_average
    print primary_metrics_average
    lb_df_average /= len(kf)
    return metrics_average, primary_metrics_average, lb_df_average


feature_name = "cogat"
labels_dir = "/home/data/nbc/athena/athena-data/feature_selection/labels/"
features_dir = "/home/data/nbc/athena/athena-data/feature_selection/features/"
out_folder = "/home/data/nbc/"
a, b, c = run_clf(feature_name, labels_dir, features_dir, out_folder)
