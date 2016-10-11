# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 2016

@author: Cody Riedel
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

#data_dir = "/Users/riedel/Desktop/athena-data/"
data_dir = "/home/data/nbc/athena/athena-data/"
predict_file = "LR1_100_best.csv"
labels_file = "test.csv"

filename_predictions = os.path.join(data_dir, "predictions_sklearn/", predict_file)
filename_labels = os.path.join(data_dir, "labels/", labels_file)
predictions = pd.read_csv(filename_predictions, header=None)
predictions = predictions.as_matrix()
labels = pd.read_csv(filename_labels)
label_names = labels.columns.values
labels = labels.as_matrix()
labels_deid = labels[: ,1:]

label_names = label_names[1:]

bd_index = np.zeros(len(label_names))
pc_index = np.zeros(len(label_names))
count = 0
for name in label_names:
    if "BehavioralDomain" in name:
        bd_index[count] = 1
    if "ParadigmClass" in name:
        pc_index[count] = 1
    count+=1

bd_locs = [i for i in range(len(bd_index)) if bd_index[i] == 1]
pc_locs = [i for i in range(len(pc_index)) if pc_index[i] == 1]

bd_labels = label_names[bd_locs]
pc_labels = label_names[pc_locs]

bd_matrix_train = labels_deid[:, bd_locs]
bd_matrix_test = predictions[:, bd_locs]
pc_matrix_train = labels_deid[:, pc_locs]
pc_matrix_test = predictions[:, pc_locs]

f1matrix_bdpc = np.zeros((len(bd_locs), len(pc_locs)))
f1matrix_bd = np.zeros((len(bd_locs),4))
for col_bd in range(bd_matrix_test.shape[1]):
    true_test_bd = [i for i in range(bd_matrix_test.shape[0]) if bd_matrix_test[i, col_bd] == 1]
    true_train_bd = [i for i in range(bd_matrix_train.shape[0]) if bd_matrix_train[i, col_bd] == 1]
    tp_bd = set(true_train_bd).intersection(true_test_bd)
    tp_bd = [i for i in tp_bd]
    if len(tp_bd) > 0:
        f1matrix_bd[col_bd,0] = f1_score(bd_matrix_train[:,col_bd], bd_matrix_test[:,col_bd])
        f1matrix_bd[col_bd,1] = precision_score(bd_matrix_train[:,col_bd], bd_matrix_test[:,col_bd])
        f1matrix_bd[col_bd,2] = recall_score(bd_matrix_train[:,col_bd], bd_matrix_test[:,col_bd])
        f1matrix_bd[col_bd,3] = hamming_loss(bd_matrix_train[:,col_bd], bd_matrix_test[:,col_bd])
        temp_pc_train = pc_matrix_train[tp_bd,:]
        temp_pc_test = pc_matrix_test[tp_bd,:]
        for col_pc in range(temp_pc_train.shape[1]):
            temp_true_test_pc = [i for i in range(temp_pc_test.shape[0]) if temp_pc_test[i,col_pc] == 1]
            temp_true_train_pc = [i for i in range(temp_pc_train.shape[0]) if temp_pc_train[i,col_pc] == 1]
            tp_pc = set(temp_true_train_pc).intersection(temp_true_test_pc)
            tp_pc = [i for i in tp_pc]
            if len(tp_pc) > 0:
                f1matrix_bdpc[col_bd, col_pc] = f1_score(temp_pc_train[:, col_pc], temp_pc_test[:, col_pc])

out_f1matrix_bd = pd.DataFrame(columns=["F1", "Precision", "Recall", "Hamming Loss"], index=bd_labels, data=f1matrix_bd)
out_f1matrix_bdpc = pd.DataFrame(columns=pc_labels, index=bd_labels, data=f1matrix_bdpc)

out_f1matrix_bd.to_csv(os.path.join(data_dir, "f1matrix_bd.csv"))
out_f1matrix_bdpc.to_csv(os.path.join(data_dir, "f1matrix_bdpc.csv"))

f1matrix_pcbd = np.zeros((len(bd_locs), len(pc_locs)))
f1matrix_pc = np.zeros((len(pc_locs),4))
for col_pc in range(pc_matrix_test.shape[1]):
    true_test_pc = [i for i in range(pc_matrix_test.shape[0]) if pc_matrix_test[i, col_pc] == 1]
    true_train_pc = [i for i in range(pc_matrix_train.shape[0]) if pc_matrix_train[i, col_pc] == 1]
    tp_pc = set(true_train_pc).intersection(true_test_pc)
    tp_pc = [i for i in tp_pc]
    if len(tp_pc) > 0:
        f1matrix_pc[col_pc,0] = f1_score(pc_matrix_train[:,col_pc], pc_matrix_test[:,col_pc])
        f1matrix_pc[col_pc,1] = precision_score(pc_matrix_train[:,col_pc], pc_matrix_test[:,col_pc])
        f1matrix_pc[col_pc,2] = recall_score(pc_matrix_train[:,col_pc], pc_matrix_test[:,col_pc])
        f1matrix_pc[col_pc,3] = hamming_loss(pc_matrix_train[:,col_pc], pc_matrix_test[:,col_pc])
        temp_bd_train = bd_matrix_train[tp_pc,:]
        temp_bd_test = bd_matrix_test[tp_pc,:]
        for col_bd in range(temp_bd_train.shape[1]):
            temp_true_test_bd = [i for i in range(temp_bd_test.shape[0]) if temp_bd_test[i, col_bd] == 1]
            temp_true_train_bd = [i for i in range(temp_bd_train.shape[0]) if temp_bd_train[i,col_bd] == 1]
            tp_bd = set(temp_true_train_bd).intersection(temp_true_test_bd)
            tp_bd = [i for i in tp_bd]
            if len(tp_bd) > 0:
                f1matrix_pcbd[col_bd, col_pc] = f1_score(temp_bd_train[:, col_bd], temp_bd_test[:, col_bd])

out_f1matrix_pc= pd.DataFrame(columns=["F1", "Precision", "Recall", "Hamming Loss"], index=pc_labels, data=f1matrix_pc)
out_f1matrix_pcbd = pd.DataFrame(columns=pc_labels, index=bd_labels, data=f1matrix_pcbd)

out_f1matrix_pc.to_csv(os.path.join(data_dir, "f1matrix_pc.csv"))
out_f1matrix_pcbd.to_csv(os.path.join(data_dir, "f1matrix_pcbd.csv"))
