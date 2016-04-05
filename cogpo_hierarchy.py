# -*- coding: utf-8 -*-
"""
Generate digraph of CogPO.

Based off this:
https://pythonhaven.wordpress.com/2009/12/09/generating_graphs_with_pydot/
"""

import os
import pydotplus as pydot
import pandas as pd


def plot(observed_cogpo, name):
    # first you create a new graph, you do that with pydot.Dot()
    graph = pydot.Dot(graph_type="graph", overlap=False)
    
    # Reduce labels to only look at Behavioral Domain, which is the only section
    # of CogPO with additional depth.
    observed_cogpo = [i[12:] for i in observed_cogpo if name in i]
    
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
    
    graph.write_png(os.path.join("/Users/salo/NBCLab/athena-data/", name+".png"))

df = pd.read_csv("/Users/salo/NBCLab/athena-data/processed_data/all_labels.csv")

observed_cogpo = df.columns[1:].astype(str).tolist()

for name in ["BehavioralDomain", "ParadigmClass"]:
    plot(observed_cogpo, name)
