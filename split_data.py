# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:34:16 2016

Determines the best way to split the training data, so that labels appear in
both datasets.

@author: salo
"""

import os
import pandas as pd
import random
import numpy as np


def resplit(labels_file, name_files):
    """
    Using existing train and test label files (to get the PMIDs) and existing
    labels file for all instances, create new train and test label files with
    new labels.
    """
    df = pd.read_csv(labels_file)
    for fi in name_files:
        df2 = pd.read_csv(fi)
        out_df = df[df['pmid'].isin(df2["pmid"])]
        out_df.to_csv(fi, index=False)


def train_test_split(labels_file, test_percent=0.33):
    """
    Find acceptable train/test data split. All labels must be represented in
    both datasets.
    """
    data_dir = os.path.dirname(labels_file)
    
    all_labels = pd.read_csv(labels_file)
    
    column_names = all_labels.columns.values
    all_data = all_labels.as_matrix()
    de_ided = all_data[:, 1:]
    
    n_instances = all_data.shape[0]
    index = range(n_instances)
    
    n_test = int(n_instances * test_percent)
    n_train = n_instances - n_test
    print("Size of test dataset: {0}".format(n_test))
    print("Size of training dataset: {0}".format(n_train))
    
    split_found = False
    while not split_found:
        shuf_index = index[:]
        random.shuffle(shuf_index)
        test_rows = de_ided[shuf_index[:n_test]]
        train_rows = de_ided[shuf_index[n_test:]]
        
        if np.all(np.sum(train_rows, axis=0)>2) and np.all(np.sum(test_rows, axis=0)>0):
            split_found = True
    
    train_data = all_data[sorted(shuf_index[n_test:]), :]
    test_data = all_data[sorted(shuf_index[:n_test]), :]
    df_train = pd.DataFrame(columns=column_names, data=train_data)
    df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    df_test = pd.DataFrame(columns=column_names, data=test_data)
    df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)


def run():
    labels_file = "/Users/salo/NBCLab/athena-data/labels/full.csv"
    train_test_split(labels_file)
