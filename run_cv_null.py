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
from sklearn.model_selection import StratifiedKFold


def unnest(lst):
    return [item for sublist in lst for item in sublist]


def _run_null(inputs):
    y_all, label_name, iter_ = inputs
    space = 'null'

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset. Classic 5x2 split.
    # 5x2 popularized in
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3325&rep=rep1&type=pdf
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=iter_)
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=iter_)

    print('\t{0}'.format(label_name))

    f_rows = []
    p_rows = []
    preds_array = np.empty((len(y_all)))
    preds_array[:] = np.NaN

    # Reduce labels and features for Context labels.
    keep_idx = np.where(np.isfinite(y_all))[0]
    y_red = y_all[keep_idx]
    red_range = np.arange(len(y_red))
    red_texts = [t for i, t in enumerate(texts) if i in keep_idx]

    for j_fold, (train_idx, test_idx) in enumerate(outer_cv.split(red_range,
                                                                  y_red)):
        print('\t\tFold {0}'.format(j_fold))

        # Get classes.
        y_train = y_red[train_idx]
        y_test = y_red[test_idx]

        most_common = stats.mode(y_train).mode[0]
        preds = y_test[:]
        preds[:] = most_common

        f_fold_label = f1_score(y_test, preds)
        f_row = ['null', 'null', 'null', j_fold, label_name, iter_,
                 f_fold_label]
        f_rows += [f_row]

        # Add new predictions to overall array.
        preds_array[keep_idx[test_idx]] = preds

    return [f_rows, preds_array, label_name]


def run(data_dir, out_dir):
    label_df = pd.read_csv(join(data_dir, 'labels/red_labels.csv'),
                           index_col='pmid')

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

    # Settings
    n_iters = 100

    # Get data from DataFrame
    pmids = label_df.index.values

    # Pull info from label_df
    label_array = label_df.as_matrix()
    label_names = label_df.columns.tolist()
    n_labels = len(label_names)

    f_alllabels = []  # One F1 array for all iters/folds/labels
    for iter_ in range(n_iters):
        # Loop for each trial
        print('Iteration {0}'.format(iter_))

        sel_params = []  # One param array for each iter
        preds_array = np.zeros(label_array.shape)  # One pred array per iter

        # Prepare inputs
        y_split = np.array_split(label_array, label_array.shape[1], axis=1)
        y_split = [y.squeeze() for y in y_split]
        iters = [iter_] * n_labels

        inputs = zip(*[y_split, label_names, iters])
        pool = mp.Pool(20)
        results = pool.map(_run_null, inputs)
        pool.close()

        f_rows, preds_1d, temp_label_names = zip(*results)
        f_alllabels += unnest(f_rows)
        for i, tl in enumerate(temp_label_names):
            idx = label_names.index(tl)
            preds_array[:, idx] = preds_1d[i]

        # Save predictions array to file.
        p_filename = 'null_null_null_{0}_preds.csv'.format(iter_)
        df = pd.DataFrame(data=preds_array, columns=label_names,
                          index=label_df.index)
        df.index.name = 'pmid'
        df.to_csv(join(out_dir, p_filename))

    # Write out [nFolds*nIters]x[nLabels] array of F1-scores to file.
    f_filename = 'null_null_null_f1.csv'
    f_cols = ['classifier', 'source', 'space', 'fold', 'label', 'iter', 'f1']

    df = pd.DataFrame(data=f_alllabels, columns=f_cols)
    df.to_csv(join(out_dir, f_filename), index=False)


if __name__ == '__main__':
    data_dir = '/home/data/nbc/athena/athena-data2/'
    out_dir = '/scratch/tsalo006/athena-cv/'
    run(data_dir, out_dir)
