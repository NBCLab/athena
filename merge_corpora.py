# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:34:05 2016
Merge the FIU and BrainMap label files.

@author: salo
"""
import os
import pandas as pd
import numpy as np
from collections import Counter

data_dir = '/Users/tsalo/Documents/nbc/athena-data/'
labels_dir = os.path.join(data_dir, 'labels/')
fiu_file = os.path.join(labels_dir, 'fiu_labels.csv')
bm_file = os.path.join(labels_dir, 'bm_labels.txt')
bmid_to_pmid_file = os.path.join(data_dir, 'misc/bmid_to_pmid.csv')

fiu_df = pd.read_csv(fiu_file)
bm_df = pd.read_csv(bm_file, sep='\t')
bmid_to_pmid_df = pd.read_csv(bmid_to_pmid_file, dtype=str)

bmid_to_pmid = dict(zip(bmid_to_pmid_df['bmid'], bmid_to_pmid_df['pmid']))
counter = Counter(bmid_to_pmid.values())
drop_pmids = []
for pmid in counter.keys():
    if counter[pmid] > 1:
        drop_pmids.append(pmid)
bmid_to_pmid = {k:v for k, v in bmid_to_pmid.items() if v not in drop_pmids}

def get2(val):
    return bmid_to_pmid.get(val, np.nan)

# Convert bmids to pmids
bmids = bm_df['bmid'].tolist()
bm_df['pmid'] = bm_df['bmid'].astype(str).apply(get2)
bm_df.drop('bmid', axis=1, inplace=True)

bm_df.dropna(axis=0, subset=['pmid'], inplace=True)
bm_df.set_index('pmid', inplace=True)

fiu_df.set_index('pmid', inplace=True)
fiu_df.index = fiu_df.index.astype(str)

# For non-Context labels, if the label doesn't occur in the FIU corpus, we
# count it as zero.
most_labels = sorted([col for col in bm_df.columns if not col.startswith('Context')])
fiu_labels = sorted(fiu_df.columns.tolist())

fiu_specific_labels = list(set(fiu_labels) - set(most_labels))
if len(fiu_specific_labels) > 0:
    print(fiu_specific_labels)

new_labels = list(set(most_labels) - set(fiu_labels))

for label in new_labels:
    fiu_df[label] = 0

# For Context labels, FIU papers get counted as NaN.
context_labels = sorted([col for col in bm_df.columns if col.startswith('Context')])
for label in context_labels:
    fiu_df[label] = np.NaN

bm_pmids = bm_df.index.tolist()
fiu_pmids = fiu_df.index.tolist()

# Use labels from BrainMap rather than FIU, when possible.
keep_fiu_pmids = list(set(fiu_pmids) - set(bm_pmids).intersection(fiu_pmids))
fiu_df = fiu_df.loc[keep_fiu_pmids]

full_df = pd.concat([bm_df, fiu_df])
full_df.index = full_df.index.astype(int)
full_df.sort_index(axis=0, inplace=True)
full_df.index = full_df.index.astype(str)

test = full_df.count(axis=0)
test_df = test.to_frame(name='total')

label_counts = full_df.sum(axis=0)
count_df = label_counts.to_frame(name='pos')
count_df = pd.concat([count_df, test_df], axis=1)
count_df['neg'] = count_df['total'] - count_df['pos']
count_df.index.name = 'label'
count_df[['pos', 'neg']].to_csv(os.path.join(labels_dir, 'label_counts.csv'))

# Both classes (pos and neg) must have at least 80 instances.
keep_df = count_df.loc[(count_df['pos'] >= 80) & (count_df['neg'] >= 80)]
keep_labels = keep_df.index

full_df.to_csv(os.path.join(labels_dir, 'untrimmed_labels.csv'))
full_df = full_df[keep_labels]
#full_df.to_csv(os.path.join(labels_dir, 'full_labels.csv'))
