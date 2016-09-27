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


def combine_features(feature_names, data_dir="/home/data/nbc/athena/athena-data/"):
    """
    Produce combined count files for selected features.
    """
    features_dir = os.path.join(data_dir, "features/")
    datasets = ["train", "test"]
    for dataset in datasets:
        path = os.path.join(features_dir, "train_")
        out_name = path + "_".join(feature_names) + ".csv"
        feature_files = [path+fn+".csv" for fn in feature_names]
        feature_dfs = [pd.read_csv(ff, dtype=float, index_col="pmid") for ff in feature_files]
        feature_df = pd.concat(feature_dfs, axis=1, ignore_index=False)
        feature_df.to_csv(out_name)
