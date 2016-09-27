# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:15:00 2016

Run sklearn classifiers to more directly compare v1 and v2

@author: tsalo006
"""
import classifier_handler_old
import os
import pandas as pd


def combine_features(feature_names, folder):
    """
    Produce combined count files for all possible combinations of features.
    """
    for dataset in ["train", "test"]:
        path = os.path.join(folder, dataset+"_")
        feature_files = [path+feature_name+".csv" for feature_name in feature_names]
        out_name = path + "_".join(feature_names) + ".csv"
        if out_name != feature_files[0]:
            feature_dfs = [[] for i in feature_names]
            for i, feature_file in enumerate(feature_files):
                feature_dfs[i] = pd.read_csv(feature_file, dtype=float)
                feature_dfs[i] = feature_dfs[i].set_index("pmid")
            feature_df = pd.concat(feature_dfs, axis=1, ignore_index=False)
            feature_df.to_csv(out_name)

data_dir = "/home/data/nbc/athena/athena-data/"
#combine_features(["nbow", "titlewords"], os.path.join(data_dir, "features"))
classifier_handler_old.run_classifiers(data_dir)
