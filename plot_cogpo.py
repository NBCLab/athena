# -*- coding: utf-8 -*-
"""
Generate digraph of CogPO.

Based off this:
https://pythonhaven.wordpress.com/2009/12/09/generating_graphs_with_pydot/
"""

from __future__ import division
import numpy as np
import seaborn as sns
import os
import pydotplus as pydot
import pandas as pd
import re


def convert_camel_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def plot_hierarchy(observed_cogpo, domain_name, out_file):
    # first you create a new graph, you do that with pydot.Dot()
    graph = pydot.Dot(graph_type="graph", overlap=False)
    
    # Reduce labels to only look at Behavioral Domain, which is the only section
    # of CogPO with additional depth.
    observed_cogpo = [i[12:] for i in observed_cogpo if domain_name in i]
    
    proc_cogpo = observed_cogpo[:]
    for label in observed_cogpo:
        sections = label.split(".")
        for i in range(1, len(sections)):
            parent = ".".join(sections[:i])
            if parent not in proc_cogpo:
                proc_cogpo += [parent]
    
    for label in proc_cogpo:
        node_name = label
        node_label = '"' + label.split(".")[-1] + '"'
        graph.add_node(pydot.Node(node_name, label=node_label))
    
    for label in proc_cogpo:
        sections = label.split(".")
        if len(sections) > 1:
            parent = ".".join(sections[:-1])
            edge = pydot.Edge(parent, label)
            graph.add_edge(edge)
    
    graph.write_png(out_file)

# Plot hierarchy
labels_file = "/home/data/nbc/athena/v1.1-data/labels/full.csv"
out_dir = "/home/data/nbc/athena/v1.1-data/figures/"

df = pd.read_csv(labels_file)
observed_cogpo = df.columns[1:].astype(str).tolist()

for domain_name in ["BehavioralDomain", "ParadigmClass"]:
    out_file = os.path.join(out_dir, convert_camel_case(domain_name)+".png")
    plot_hierarchy(observed_cogpo, domain_name, out_file)

# Plot heatmap
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
fig.savefig(os.path.join(out_dir, "cogpo_correlations.png"))
