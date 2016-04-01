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
    df_train.to_csv(os.path.join(data_dir, "train_labels.csv"), index=False)
    df_test = pd.DataFrame(columns=column_names, data=test_data)
    df_test.to_csv(os.path.join(data_dir, "test_labels.csv"), index=False)


def run():
    labels_file = "/Users/salo/NBCLab/athena-data/processed_data/all_labels.csv"
    train_test_split(labels_file)
