# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:51:18 2016

Plot correlation between f-score from BR classifier and label count.

@author: tsalo006
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
import numpy as np

df = pd.read_csv("/home/data/nbc/athena/athena-data/statistics/BR_with_count.csv")
df = df[["Count", "F1"]]
df.columns = ["Count", "F-Score"]

corr, p = stats.pearsonr(df["Count"], df["F-Score"])
if p < 0.001:
    p_str = "p < 0.001"
elif p < 0.05:
    p_str = "p < 0.05"
else:
    p_str = "p = {:.4f}".format(p)

sns.set_style("darkgrid")
ax = sns.regplot(x="Count", y="F-Score", x_estimator=np.mean, data=df, x_ci=None,
                     ci=99, color="darkblue", scatter=True, robust=True)
                 # robust takes too long for anything short of the real deal
sns.plt.title("Relationship Between Label Count and F-Score")
sns.plt.xlim((0, 850))
sns.plt.ylim((0, 1))
ax.annotate(r"$r^2$" + " = {:.4f}\n{}".format(corr**2, p_str), xy=(.76, .25), xycoords=ax.transAxes,
            fontsize=14, bbox=dict(facecolor="white", edgecolor="black", pad=4.0))

sns.plt.savefig("/home/data/nbc/athena/athena-data/figures/count_fscore_corr.png",
                dpi=400)
