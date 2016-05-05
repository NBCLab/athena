# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:44:36 2016

Plot domain-wise comparison of F-scores between v1.1 and v2.0.

@author: salo
"""

import seaborn as sns
import pandas as pd

v1_results = "/Users/salo/NBCLab/athena-data/feature_selection/results/authoryear_labelwise.csv"
v2_results = "/Users/salo/NBCLab/athena-data/feature_selection/results/cogat_labelwise.csv"

v1_df = pd.read_csv(v1_results)
v1_df = v1_df[["Label", "F1"]]
v1_df.columns = ["Label", "V1"]
v2_df = pd.read_csv(v2_results)
v2_df = v2_df[["Label", "F1"]]
v2_df.columns = ["Label", "V2"]

df = pd.merge(v1_df, v2_df, on=["Label"])
j = df["Label"].values
j = [".".join(i.split(".")[1:]) for i in j]
domains = [i.split(".")[0] for i in j]
j = [".".join(i.split(".")[1:]) for i in j]
df["Label"] = j
df["Domain"] = domains

df2 = pd.melt(df, id_vars=["Label", "Domain"], value_vars=["V1", "V2"], var_name="Version", value_name="F-score")

ax = sns.barplot(x="Domain", y="F-score", hue="Version", units="Label", data=df2, ci=95, n_boot=1000)
sns.plt.title("F-Scores Between ATHENA Versions")
fig = ax.get_figure()
fig.savefig("version_fscores.png", dpi=400)
