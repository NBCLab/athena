# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:34:16 2016

Determines the best way to split the training data, so that labels appear in
both datasets.

@author: salo
"""

import pandas as pd
import random
import numpy as np

labels_file = "/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv"
train_labels = pd.read_csv(labels_file)

labels = train_labels.as_matrix()
de_ided = labels[:, 1:]

n_instances = labels.shape[0]
index = range(n_instances)

n_test = int(n_instances * .33)
n_train = n_instances - n_test

contd = True
while contd:
    shuf_index = index[:]
    random.shuffle(shuf_index)
    test_rows = de_ided[shuf_index[:n_test]]
    train_rows = de_ided[shuf_index[n_test:]]
    
    if np.all(np.sum(train_rows, axis=0)>2) and np.all(np.sum(test_rows, axis=0)>0):
        contd = False

