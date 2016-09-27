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
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from collections import OrderedDict
import evaluate_classifier as ec


def run_classifiers(data_dir="/home/data/nbc/athena/athena-data/"):
    """
    Run sklearn classifiers.
    """
    features_dir = os.path.join(data_dir, "features/")
    labels_dir = os.path.join(data_dir, "labels/")
    stats_dir = os.path.join(data_dir, "statistics_sklearn/")
    
    # Train
    train_features_file = os.path.join(features_dir, "train_nbow.csv")
    test_features_file = os.path.join(features_dir, "test_nbow.csv")
    train_labels_file = os.path.join(labels_dir, "train.csv")
    test_labels_file = os.path.join(labels_dir, "test.csv")
    
    train_features_df = pd.read_csv(train_features_file)
    train_features = train_features_df.as_matrix()[:, 1:]
    test_features_df = pd.read_csv(test_features_file)
    test_features = test_features_df.as_matrix()[:, 1:]

    train_labels_df = pd.read_csv(train_labels_file)
    train_labels = train_labels_df.as_matrix()[:, 1:]
    test_labels_df = pd.read_csv(test_labels_file)

    classifiers = OrderedDict([("MNB", MultinomialNB()),
                               ("BNB", BernoulliNB()),
                    		    ("LR1", LogisticRegression(penalty="l1", class_weight="auto")),
                    		    ("LR2", LogisticRegression(penalty="l2", class_weight="auto")), 
                    		    ("SVC1", LinearSVC(penalty="l1", class_weight="auto", dual=False)),
                    		    ("SVC2", LinearSVC(penalty="l2", class_weight="auto", dual=False))])
    parameters = [
    		{'estimator__alpha': [0.01, 0.1, 1, 10]},
    		{'estimator__alpha': [0.01, 0.1, 1, 10]}, 
    		{'estimator__C': [0.1, 1, 10, 100]},
    		{'estimator__C': [0.01, 0.1, 1, 10]},
    		{'estimator__C': [0.01, 0.1, 1, 10]},
    		{'estimator__C': [0.01, 0.1, 1, 10]}]
    		
    original_params = [
		{'estimator__alpha': [0.1]},
		{'estimator__alpha': [0.1]}, 
		{'estimator__C': [100]},
		{'estimator__C': [10]},
		{'estimator__C': [10]},
		{'estimator__C': [1]}]
	
    out_metrics = []
    out_primary_metrics = []
    for i, clf in enumerate(classifiers):
        # BEST
        grid = GridSearchCV(OneVsRestClassifier(classifiers[clf]), parameters[i],
                            cv=KFold(train_features.shape[0], n_folds=10, shuffle=True),
                            scoring="f1_macro", verbose=1)
        grid.fit(train_features, train_labels)
        parameter = grid.best_params_[parameters[i].keys()[0]]
        
        if clf in ["MNB", "BNB"]:
            classifiers[clf].set_params(alpha=parameter)
        else:
            classifiers[clf].set_params(C=parameter)
        
        classif = OneVsRestClassifier(classifiers[clf])
        classif.fit(train_features, train_labels)
        model_name = classifiers.keys()[i]+"_"+str(parameter).replace(".", "_")+"_best"
        metrics, primary = test_eval(classif, test_features,
                                     test_labels_df, model_name, data_dir)
        metrics.insert(0, model_name)
        primary.insert(0, model_name)
        out_metrics += [metrics]
        out_primary_metrics += [primary]
        
        # OLD
        parameter = original_params[i][original_params[i].keys()[0]][0]
        if clf in ["MNB", "BNB"]:
            classifiers[clf].set_params(alpha=parameter)
        else:
            classifiers[clf].set_params(C=parameter)
        classif = OneVsRestClassifier(classifiers[clf])
        classif.fit(train_features, train_labels)
        
        model_name = classifiers.keys()[i]+"_"+str(parameter).replace(".", "_")+"_old"
        metrics, primary = test_eval(classif, test_features,
                                     test_labels_df, model_name, data_dir)
        metrics.insert(0, model_name)
        primary.insert(0, model_name)
        out_metrics += [metrics]
        out_primary_metrics += [primary]
    out_df = pd.DataFrame(columns=["Model", "Macro F1", "Micro F1",
                                   "Macro Precision", "Micro Precision",
                                   "Macro Recall", "Micro Recall",
                                   "Hamming Loss"], data=out_metrics)
    out_df.to_csv(os.path.join(stats_dir, "results.csv"), index=False)
    primary_df = pd.DataFrame(columns=["Model", "Macro F1", "Micro F1",
                                       "Macro Precision", "Micro Precision",
                                       "Macro Recall", "Micro Recall",
                                       "Hamming Loss"], data=out_primary_metrics)
    primary_df.to_csv(os.path.join(stats_dir, "primary_results.csv"), index=False)


def test_eval(classif, test_features, test_labels_df, name, type_dir):
    test_labels = test_labels_df.as_matrix()[:, 1:]
    label_names = test_labels_df.columns

    # Test
    preds_dir = os.path.join(type_dir, "predictions_sklearn/")
    predictions = classif.predict(test_features)
    out_file = os.path.join(preds_dir, "{}.csv".format(name))
    np.savetxt(out_file, predictions, delimiter=",")
    
    # Evaluate
    stats_dir = os.path.join(type_dir, "statistics_sklearn/")
    metrics = ec.return_metrics(test_labels, predictions)
    primary_metrics = ec.return_primary(test_labels, predictions, label_names)
    lb_df = ec.return_labelwise(test_labels_df, predictions)
    lb_df.set_index("Label", inplace=True)
    lb_df.to_csv(os.path.join(stats_dir, name+"_labelwise.csv"))
    return metrics, primary_metrics
