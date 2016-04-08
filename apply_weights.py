# -*- coding: utf-8 -*-
"""
Untar files and organize by site and study.
"""

# NOTES
"""
761 * 1721 = 1309681
zeros =      1297515

"""

import pandas as pd
from cogat import apply_weights_recursively

input_df = pd.read_csv("/Users/salo/NBCLab/athena-data/features/cogat_res.csv")

weight_files = ["/Users/salo/NBCLab/athena-data/gazetteers/cogat_weights_ws2_up_20160406.csv",
                "/Users/salo/NBCLab/athena-data/gazetteers/cogat_weights_ws2_down_20160406.csv",
                "/Users/salo/NBCLab/athena-data/gazetteers/cogat_weights_ws2_side_20160406.csv"]
weight_dfs = [pd.read_csv(file_) for file_ in weight_files]

# Some terms may no longer exist, but may still be referenced in assertions.
# As such, the weight_df will likely be larger than the input_df
# To fix this, columns of zeros will be added to the input_df 
for i in range(len(weight_dfs)):
    weight_dfs[i] = weight_dfs[i].set_index("Unnamed: 0")
input_df.set_index("pmid")
cogat_input = input_df.columns.values.tolist()

weight_df = weight_dfs[0]
cogat_weight = weight_df.columns.values.tolist()

not_in_input = list(set(cogat_weight) - set(cogat_input))

add = [term for term in not_in_input if term.startswith("ctp")]
drop = [term for term in not_in_input if not term.startswith("ctp")]

new_weight = sorted(list(set(cogat_weight) - set(drop)))

for i in range(len(weight_dfs)):
    weight_dfs[i] = weight_dfs[i][new_weight]
    weight_dfs[i] = weight_dfs[i][weight_dfs[i].index.isin(new_weight)]

# Add to input
for val in add:
    input_df[val] = 0

input_df = input_df[new_weight]

for i in weight_dfs:
    print i.shape
print input_df.shape

aa = apply_weights_recursively(input_df, weight_dfs=weight_dfs)
aa.to_csv("WEIGHTED.csv")
