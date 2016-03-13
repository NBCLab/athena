# -*- coding: utf-8 -*-
"""
Generate digraph of CogPO.

"""

import pandas as pd


def gen_string(dict_, tabs, in_string):
    for key in sorted(dict_.keys()):
        in_string += '\n{0}<label name="{1}"></label>'.format(tabs, key)
        if dict_[key]:
            in_tabs = tabs
            in_tabs += "\t"
            in_string = gen_string(dict_[key], in_tabs, in_string)
            in_string += '\n{0}</label>'.format(tabs)
    return in_string

train_labels = "/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv"

df = pd.read_csv(train_labels)
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

with open("/Users/salo/NBCLab/athena-data/processed_data/label_hierarchy.xml", "w") as fo:
    fo.write(out_string)
