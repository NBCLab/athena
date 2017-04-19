# -*- coding: utf-8 -*-
"""
Compare classifiers, feature spaces, and feature sources.

Only the source comparison has been written so far.
"""
from glob import glob
from os.path import join, basename
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare

comparisons = ['source', 'space', 'classifier', 'fold', 'label']
comparers = {'source': ['full', 'abstract'],
             'space': ['bow', 'cogat'],
             'classifier': ['svm', 'knn', 'lr', 'bnb']}
patterns = {'source': '*_{0}_*_f1.csv',
            'space': '*_*_{0}_f1.csv',
            'classifier': '{0}_*_*_f1.csv'}

f1s_dir = '/scratch/tsalo006/test_cv/'


def minus(l, s):
    """
    Remove an element from a list without compromising the original object.
    """
    return [el for el in l if el != s]


def run_the_thing(data):
    k = len(data)
    N = k * len(data[0])
    
    df1 = k - 1
    df2 = N - 1
    
    # Repeated measures one-way ANOVA?
    
    f, p = friedmanchisquare(*split_data)  # Currently independent and won't run.
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


# Compare sources
print('Comparing sources.')

for comp in sorted(patterns.keys()):
    merged_dfs = []
    var = comparers[comp]
    for v in var:
        files = glob(join(f1s_dir, patterns[comp].format(v)))
        dfs = []
        for f in files:
            combo_df = pd.read_csv(f)
            # Fix mislabeling
            combo_df = combo_df.rename(columns={'label': 'fold', 'fold': 'label'})
            combo_df['fold'] = combo_df['fold'].astype(str)
    
            # Create unique identifier for specific fold/space/clf/etc combo.
            # We specified our random seeds in the CV so the split for a given
            # combo should be constant across sources.
            cols = minus(comparisons, comp)
            combo_df['id'] = combo_df[cols].apply(lambda x: '-'.join(x), axis=1)
            combo_df.set_index('id', inplace=True)
    
            # Remove unnecessary columns and give score column unique name.
            combo_df[v] = combo_df['f1']
            combo_df = combo_df[[v]]
            dfs.append(combo_df)
        full_df = pd.concat(dfs, ignore_index=False)
        full_df.sort_index(inplace=True)
        merged_dfs.append(full_df)

    # Create (n folds * n spaces * n clfs * n boosted * n labels) X (n sources) df
    merged_df = pd.concat(merged_dfs, axis=1, join='inner')
    size_check = [merged_df.shape[0]==df.shape[0] for df in merged_dfs]
    if not all(size_check):
        raise Exception('Merged df size is bad: {0}'.format(merged_df.shape))
    
    source_data = merged_df.as_matrix()
    split_data = np.split(source_data, source_data.shape[1], axis=1)
    run_the_thing(split_data)
