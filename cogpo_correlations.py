# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 18:39:31 2016
Get fraction of studies positive for a given CogPO label that are also positive
for another label. Produces a directed graph.
@author: salo
"""
from __future__ import division
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("/Users/salo/NBCLab/athena-data/processed_data/train_labels.csv")

columns = df.columns[1:].tolist()
data = df.values[:, 1:]

out = np.zeros((data.shape[1], data.shape[1]))
for i in range(data.shape[1]):
    for j in range(data.shape[1]):
        pos_i = np.where(data[:, i]==1)[0]
        pos_j = np.where(data[:, j]==1)[0]
        diff = np.setdiff1d(pos_j, pos_i).shape[0]
        if pos_j.shape[0] == 0:
            frac = 0
        else:
            frac = 1 - (diff / pos_j.shape[0])
        out[i, j] = frac

cmap = sns.diverging_palette(220, 10, as_cmap=True)
plot = sns.heatmap(out, cmap=cmap, xticklabels=False, yticklabels=False)
fig = plot.get_figure()
fig.savefig("/Users/salo/NBCLab/athena-data/heatmap.png")
