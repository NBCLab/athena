# -*- coding: utf-8 -*-
"""
Compare classifiers, feature spaces, and feature sources.

Only the source comparison has been written so far.
"""
from glob import glob
from os.path import join
import pandas as pd
import numpy as np
from scipy.stats import f_oneway as anova

comparisons = ['source', 'space', 'classifier', 'fold', 'label']
sources = ['full', 'abstract', 'methods', 'combined']
spaces = ['bow', 'cogat']
classifiers = ['svm', 'knn', 'lr', 'nb']

f1s_dir = '/Users/salo/Desktop/'

def minus(l, s):
    """
    Remove an element from a list without compromising the original object.
    """
    return [el for el in l if el != s]

# Compare sources
print('Comparing sources.')

pattern = '*_{0}_*_f1.txt'

merged_dfs = []
for source in sources:
    files = glob(join(f1s_dir, pattern.format(source)))
    dfs = []
    for f in files:
        combo_df = pd.read_csv(f)
        
        # Create unique identifier for specific fold/space/clf/etc combo.
        # We specified our random seeds in the CV so the split for a given
        # combo should be constant across sources.
        cols = minus(comparisons, 'source')
        combo_df['id'] = combo_df[cols].apply(lambda x: '-'.join(x), axis=1)
        combo_df.set_index('id', inplace=True)
        
        # Remove unnecessary columns and give score column unique name.
        combo_df[source] = combo_df['f1']
        combo_df = combo_df[[source]]
        dfs.append(combo_df)
    source_df = pd.concat(dfs, ignore_index=False)
    source_df.sort_index(inplace=True)
    merged_dfs.append(source_df)

# Create (n folds * n spaces * n clfs * n boosted * n labels) X (n sources) df
merged_df = pd.concat(merged_dfs, axis=1, join='inner')
size_check = [merged_df.shape==df.shape for df in merged_dfs]
if not all(size_check):
    raise Exception('Merged df size is bad: {0}'.format(merged_df.shape))

source_data = merged_df.as_matrix()
k = source_data.shape[1]
N = np.prod(source_data.shape)

df1 = k - 1
df2 = N - 1

# Repeated measures one-way ANOVA?
f, p = anova(source_data)  # Currently independent and won't run.
if p < 0.05:  # Currently no MCC.
    res = ''
    run_posthocs = True
else:
    res = 'not '
    run_posthocs = False

print('An analysis of variance showed that the effect of feature source '
      'on F1-score was {res}significant, F({df1}, {df2}) = {F}, p = {p}.'.format(df1=df1,
                                                                                 df2=df2,
                                                                                 F=f, p=p,
                                                                                 res=res))

if run_posthocs:
    print('Running posthoc tests.')
