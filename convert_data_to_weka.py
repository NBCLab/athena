# -*- coding: utf-8 -*-
"""
Prepare data for classification using MEKA and MULAN.
"""

import pandas as pd
import os


def convert_to_arff(feature_files,
                    label_file="/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv"):
    """
    Convert pandas DataFrames to arff format. Allows user to use combinations
    of features.
    """
    out_dir = os.path.dirname(label_file)
    
    # Load and combine data
    out_name = "train_data"
    feature_dfs = [[] for i in feature_files]
    for i, feature_file in enumerate(feature_files):
        file_ = os.path.basename(feature_file)
        feature_name = "_" + file_.split("features_")[-1].split(".csv")[0]
        out_name += feature_name
        
        feature_dfs[i] = pd.read_csv(feature_file, dtype=float)
        feature_dfs[i] = feature_dfs[i].set_index("pmid")
    out_name += ".arff"
    
    feature_df = pd.concat(feature_dfs, axis=1, ignore_index=False)
    features = feature_df.columns.tolist()
    
    label_df = pd.read_csv(label_file, dtype=int)
    label_df = label_df.set_index("pmid")
    labels = label_df.columns.tolist()
    
    out_string = "@relation TrainingData\n"
    for feature in features:
        if " " in feature:
            out_string += '@attribute "{0}" numeric\n'.format(feature)
        else:
            out_string += '@attribute {0} numeric\n'.format(feature)
    
    for label in labels:
        if " " in label:
            out_string += '@attribute "{0}" {{0, 1}}\n'.format(label)
        else:
            out_string += '@attribute {0} {{0, 1}}\n'.format(label)
    out_string += "\n@data\n"
    
    for pmid in label_df.index.values:
        feature_list = feature_df.loc[pmid].tolist()
        feature_str = ",".join(map(str, feature_list))
        
        label_list = label_df.loc[pmid].tolist()
        label_str = ",".join(map(str, label_list))
        
        out_string += "{0},{1}\n".format(feature_str, label_str)
    
    out_file = os.path.join(out_dir, out_name)
    with open(out_file, "w") as fo:
        fo.write(out_string)


def gen_string(dict_, tabs, in_string):
    """
    Used by gen_hier_label_file.
    """
    for key in sorted(dict_.keys()):
        in_string += '\n{0}<label name="{1}"></label>'.format(tabs, key)
        if dict_[key]:
            in_tabs = tabs
            in_tabs += "\t"
            in_string = gen_string(dict_[key], in_tabs, in_string)
            in_string += '\n{0}</label>'.format(tabs)
    return in_string


def gen_hier_label_file(label_file="/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv"):
    """
    Creates MULAN-format XML file to specify hierarchical labels.
    """
    out_dir = os.path.dirname(label_file)
    out_file = os.path.join(out_dir, "label_hierarchy.xml")
    
    df = pd.read_csv(label_file)
    labels = df.columns.tolist()[1:]
    
    label_hierarchy = {}
    
    for label in labels:
        label_components = label.split(".")
        local_result = label_hierarchy
        for i in range(2, len(label_components)):
            string = ".".join(label_components[:i+1])
            local_result = local_result.setdefault(string, {})
    
    out_string = '<labels xmlns="http://mulan.sourceforge.net/labels">'
    out_string = gen_string(label_hierarchy, "\t", out_string)
    out_string += '\n</labels>'
    
    with open(out_file, "w") as fo:
        fo.write(out_string)


def test():
    feature_files = ["/Users/salo/NBCLab/athena-data/processed_data/train_features_ay.csv",
                     "/Users/salo/NBCLab/athena-data/processed_data/train_features_j.csv"]
    convert_to_arff(feature_files)
    gen_hier_label_file()
