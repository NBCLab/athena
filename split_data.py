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
ids = labels[:, 0]

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

train_ids = ids[shuf_index[n_test:]]
test_ids = ids[shuf_index[:n_test]]
df = pd.DataFrame(columns=["PubMed ID"], data=sorted(train_ids))
df.to_csv("/Users/salo/NBCLab/athena-data/processed_data/train_ids.csv", index=False)
df2 = pd.DataFrame(columns=["PubMed ID"], data=sorted(test_ids))
df2.to_csv("/Users/salo/NBCLab/athena-data/processed_data/test_ids.csv", index=False)
