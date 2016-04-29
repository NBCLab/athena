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

def run_classifiers(data_dir="/home/data/nbc/athena/v1.1-data/"):
    """
    Run sklearn OneVsRest multilabel classifier wrapped around a LinearSVC
    binary classifier with l2 penalty and 1.0 C on a given feature count file.
    
    Cross-validation is used to evaluate results.
    """
    feature_name = "nbow"
    features_dir = "/home/data/nbc/athena/v1.1-data/features/"
    labels_dir = "/home/data/nbc/athena/v1.1-data/labels/"
    
    # Train
    train_features_file = os.path.join(features_dir, "train_"+feature_name+".csv")
    test_features_file = os.path.join(features_dir, "test_"+feature_name+".csv")
    labels_file = os.path.join(labels_dir, "train.csv")
    
    train_features_df = pd.read_csv(train_features_file)
    train_features = train_features_df.as_matrix()[:, 1:]
    test_features_df = pd.read_csv(test_features_file)
    test_features = test_features_df.as_matrix()[:, 1:]
    
    train_labels_df = pd.read_csv(train_labels_file)
    train_labels = train_labels_df.as_matrix()[:, 1:]
    test_labels_df = pd.read_csv(test_labels_file)
    test_labels = test_labels_df.as_matrix()[:, 1:]
    
    label_names = train_labels_df.columns
    
    classifiers = {"MNB": MultinomialNB(),
        		   "BNB": BernoulliNB(),
        		   "LR1": LogisticRegression(penalty = "l1", class_weight="auto"),
        		   "LR2": LogisticRegression(penalty = "l2", class_weight="auto"), 
        		   "SVC1": LinearSVC(penalty = "l1", class_weight="auto", dual=False),
        		   "SVC2": LinearSVC(penalty = "l2", class_weight="auto", dual=False)]
        		   }
    parameters = [
		{'ovr__estimator__alpha':[0.01, 0.1, 1, 10]},
		{'ovr__estimator__alpha':[0.01, 0.1, 1, 10]}, 
		{'ovr__estimator__C':[0.1, 1, 10, 100]},
		{'ovr__estimator__C':[0.01, 0.1, 1, 10]},
		{'ovr__estimator__C':[0.01, 0.1, 1, 10]},
		{'ovr__estimator__C':[0.01, 0.1, 1, 10]}]
		
	original_params = [
		{'ovr__estimator__alpha':[0.1]},
		{'ovr__estimator__alpha':[0.1]}, 
		{'ovr__estimator__C':[100]},
		{'ovr__estimator__C':[10]},
		{'ovr__estimator__C':[10]},
		{'ovr__estimator__C':[1]}]
		
    for i, clf in enumerate(classifiers):
        ## POST HERE IS FROM FEATURE SELECTION (NOT ADAPTED)
        ## PERFORM GRID SEARCH HERE
        grid = GridSearchCV(clf, parameters[i], cv = KFold(len(train_features), n_folds = 10, shuffle=True), scoring = "f1_micro", verbose = 1)
        grid.fit(train_features, train_labels)
        parameter = grid.best_params_[parameters[i].keys()[0]]
        best_estimator = grid.best_estimator_
        
        clf.set_params(parameters[i].keys()[0], parameter)
        
        classif = OneVsRestClassifier(clf)
        classif.fit(train_features, train_labels)
        name = classifiers.keys()[i]+"_"+parameter+"_new"
        test_eval(classif, test_features, label_names, test_labels, test_labels_df, name)
        
        parameter = original_params[i][original_params[i].keys()[0]][0]
        clf.set_params(original_params[i].keys()[0], parameter)
        classif.fit(train_features, train_labels)
        
        name = classifiers.keys()[i]+"_"+parameter+"_old"
        test_eval(classif, test_features, label_names, test_labels, test_labels_df, name)

def test_eval(classif, test_features, label_names, test_labels, test_labels_df, name):
    # Test
    predictions = classif.predict(test_features)
    out_file = os.path.join(out_folder, "{}_k{}.csv".format(name, k))
    np.savetxt(out_file, predictions, delimiter=",")
    
    # Evaluate
    metrics = ec.return_metrics(test_labels, predictions)
    primary_metrics = ec.return_primary(test_labels, predictions, label_names)
    lb_df = ec.return_labelwise(test_labels_df, predictions)
    lb_df.set_index("Label", inplace=True)
    
    primary_out_file = os.path.join(out_folder, "{}_k{}.csv".format(name+"_primary_metrics", k))
    np.savetxt(primary_out_file, primary_metrics, delimiter=",")
    
    metrics_out_file = os.path.join(out_folder, "{}_k{}.csv".format(name+"_metrics", k))
    np.savetxt(metrics_out_file, metrics, delimiter=",")
    
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


def dataset_statistics(data_dir="/home/data/nbc/athena/v1.1-data/", feature_name="nbow"):
    for text_type in ["full", "combined"]:
        type_dir = os.path.join(data_dir, text_type)
        labels_dir = os.path.join(type_dir, "labels")
        features_dir = os.path.join(type_dir, "features")
        statistics_file = os.path.join(type_dir, "statistics/dataset_statistics.csv")
        
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
