"""
Run a null version of the cross validation procedure.
"""
from glob import glob
import cPickle as pickle
import multiprocessing as mp
from os.path import basename, splitext, join

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def unnest(lst):
    return [item for sublist in lst for item in sublist]


def run(data_dir, out_dir):
    label_df = pd.read_csv(join(data_dir, 'labels/red_labels.csv'),
                           index_col='pmid')
    sources = ['full', 'abstract']

    # BOW PMIDs
    text_dirs = [join(data_dir, 'text/stemmed_{0}/'.format(s))
                 for s in sources]
    texts = [glob(join(td, '*.txt')) for td in text_dirs]
    bow_pmids_ = [[int(splitext(basename(f))[0]) for f in t] for t in texts]
    bow_pmids = sorted(list(set(bow_pmids_[0]).intersection(bow_pmids_[1])))

    # CogAt PMIDs
    cogat_dfs = [pd.read_csv(join(data_dir,
                                  'features/cogat_{0}.csv'.format(s)),
                             index_col='pmid') for s in sources]
    cogat_pmids_ = [df.index.tolist() for df in cogat_dfs]
    cogat_pmids = set(cogat_pmids_[0]).intersection(cogat_pmids_[1])
    cogat_pmids = sorted(list(cogat_pmids))

    # Label PMIDs
    label_pmids = label_df.index.tolist()

    # Intersection between all three sets
    shared_pmids = set(label_pmids).intersection(bow_pmids)
    shared_pmids = shared_pmids.intersection(cogat_pmids)
    shared_pmids = sorted(list(shared_pmids))

    # Reduce corpus by PMIDs with features and labels
    label_df = label_df.loc[shared_pmids]
    pred_label_df = label_df.copy()

    # Use label cardinality and most popular labels to generate predicted labels
    labels = label_df.columns.tolist()
    dimensions = sorted(list(set([l.split('.')[0] for l in labels])))
    for dim in dimensions:
        dim_labels = [l for l in labels if l.startswith(dim+'.')]
        dim_df = label_df[dim_labels]
        lab_card = dim_df.sum(axis=1).mean()
        lab_card = int(np.ceil(lab_card))
        chosen = dim_df.sum(axis=0).sort_values(ascending=False)[:lab_card]
        chosen = chosen.index.tolist()
        pred_dim_df = dim_df.copy()
        pred_dim_df.loc[:, :] /= 2.
        for lab in pred_dim_df.columns:
            pred_label_df[lab] = pred_dim_df[lab]

        pred_dim_df = np.floor(pred_dim_df.copy())
        pred_dim_df[chosen] += 1
        for chos in chosen:
            pred_label_df[chos] = pred_dim_df[chos]

    # Get data from DataFrame
    pmids = label_df.index.values

    # Pull info from label_df
    label_array = label_df.values
    label_names = label_df.columns.tolist()
    pred_label_array = pred_label_df[label_names].values
    n_labels = len(label_names)

    f_rows = []  # One F1 array for all labels

    # Prepare inputs
    y_split = np.array_split(label_array, label_array.shape[1], axis=1)
    y_split = [y.squeeze() for y in y_split]
    y_pred_split = np.array_split(pred_label_array, label_array.shape[1], axis=1)
    y_pred_split = [y.squeeze() for y in y_pred_split]

    for label in range(n_labels):
        y_label = y_split[label]
        y_pred_label = y_pred_split[label]
        y_pred_label = y_pred_label[~np.isnan(y_label)].astype(int)
        y_label = y_label[~np.isnan(y_label)].astype(int)
        f_label = f1_score(y_label, y_pred_label)
        f_row = [label_names[label], f_label]
        f_rows += [f_row]

    # Write out [nFolds*nIters]x[nLabels] array of F1-scores to file.
    f_filename = 'null_f1.csv'
    f_cols = ['label', 'f1']

    df = pd.DataFrame(data=f_rows, columns=f_cols)
    df.to_csv(join(out_dir, f_filename), index=False)


if __name__ == '__main__':
    data_dir = '/home/data/nbc/athena/athena/athena-data2/'
    out_dir = '/home/data/nbc/athena/athena/athena-cv-null/'
    run(data_dir, out_dir)
