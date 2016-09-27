# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:08:47 2016

Try evaluating the classifiers.

@author: tsalo006
"""

import numpy as np
import itertools
import pandas as pd
import os
import evaluate_classifier as ec
import copy
import shutil
from glob import glob

labels_dir = "/home/data/nbc/athena/athena-data/labels/"
pred_dir = "/home/data/nbc/athena/athena-data/predictions/"
stats_dir = "/home/data/nbc/athena/athena-data/statistics/"

labels_file = os.path.join(labels_dir, "test.csv")
pred_files = glob(os.path.join(pred_dir, "*.csv"))

out_metrics = []
out_primary_metrics = []
for pred_file in pred_files:
    model_name = os.path.basename(os.path.splitext(pred_file)[0])
    model_name = model_name.split(".")[-1][:-1]
    predictions = np.loadtxt(pred_file, delimiter=",")

    # Evaluate
    metrics = ec.return_metrics(labels_file, predictions)
    metrics.insert(0, model_name)
    primary_metrics = ec.return_primary(labels_file, predictions)
    primary_metrics.insert(0, model_name)
    lb_df = ec.return_labelwise(labels_file, predictions)
    lb_df.set_index("Label", inplace=True)
    lb_df.to_csv(os.path.join(stats_dir, model_name+".csv"))

    out_metrics += [metrics]
    out_primary_metrics += [primary_metrics]

out_df = pd.DataFrame(columns=["Model", "Macro F1", "Micro F1",
                               "Macro Precision", "Micro Precision",
                               "Macro Recall", "Micro Recall",
                               "Hamming Loss"], data=out_metrics)
out_df.to_csv(os.path.join(stats_dir, "metrics.csv"), index=False)

pri_df = pd.DataFrame(columns=["Model", "Macro F1", "Micro F1",
                               "Macro Precision", "Micro Precision",
                               "Macro Recall", "Micro Recall",
                               "Hamming Loss"], data=out_primary_metrics)
pri_df.to_csv(os.path.join(stats_dir, "primary_metrics.csv"), index=False)
