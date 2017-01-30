# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:46:14 2017
Reduce labels file based on available text files.
@author: tsalo006
"""
import os
import pandas as pd
from glob import glob

data_dir = '/home/data/nbc/athena/athena-data2/'
text_dir = os.path.join(data_dir, 'text')
labels_dir = os.path.join(data_dir, 'labels')

labels_file = os.path.join(labels_dir, 'full_labels.csv')
out_file = os.path.join(labels_dir, 'red_labels.csv')
out_texts_file = os.path.join(text_dir, 'texts.csv')

label_df = pd.read_csv(labels_file, index_col='pmid')

# Reduce by data availability
available_abstracts = glob(os.path.join(text_dir, 'cleaned_abstract/*.txt'))
available_fulltexts = glob(os.path.join(text_dir, 'cleaned_full/*.txt'))

available_abstracts = [os.path.basename(os.path.splitext(f)[0]) for f in available_abstracts]
available_fulltexts = [os.path.basename(os.path.splitext(f)[0]) for f in available_fulltexts]

available_texts = list(set(available_abstracts).intersection(available_fulltexts))
available_texts = sorted([int(f) for f in available_texts])

text_df = pd.DataFrame(columns=['abstract', 'full'])
text_df.index.name = 'pmid'
for pmid in available_texts:
    pmid_texts = ['', '']
    with open(os.path.join(text_dir, 'cleaned_abstract/{0}.txt'.format(pmid)), 'r') as fo:
        pmid_texts[0] = fo.read()
    
    with open(os.path.join(text_dir, 'cleaned_full/{0}.txt'.format(pmid)), 'r') as fo:
        pmid_texts[1] = fo.read()
    text_df.loc[pmid] = pmid_texts

text_df.to_csv(out_texts_file, index_label='pmid')

label_df = label_df.loc[available_texts]

# Reduce by label counts
min_ = 50

# Remove labels with too few positive instances
label_counts = label_df.sum()
keep_labels = label_counts[label_counts>=min_].index
label_df = label_df[keep_labels]

# Remove labels with too few negative instances
label_counts = label_df.shape[0] - label_df.sum()
keep_labels = label_counts[label_counts>=min_].index
label_df = label_df[keep_labels]
label_df.to_csv(out_file, index_label='pmid')
