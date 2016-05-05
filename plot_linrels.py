# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:18:43 2016

Plot linear relationships between F-score and label representation (count) 
between domains.

@author: salo
"""

import seaborn as sns
import pandas as pd

results = "/Users/salo/Documents/BR_with_count.csv"

df = pd.read_csv(results)

df = df[["Label", "Count", "F1"]]

j = df["Label"].values
j = [".".join(i.split(".")[1:]) for i in j]
domains = [i.split(".")[0] for i in j]
j = [".".join(i.split(".")[1:]) for i in j]
df["Label"] = j
df["Domain"] = domains

ax = sns.lmplot(x="Count", y="F1", hue="Domain", units="Label", data=df)
sns.plt.xlim((0, 1000))
sns.plt.ylim((0, 1))

sns.plt.title("Relationship Between F-Score and Count")
ax.savefig("domain_count_relationships.png", dpi=400)