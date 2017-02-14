""" Example of how we'll be performing CV in ATHENA paper.
Set for CogAt currently (with Iris as the example dataset).

1) Loop through random trials
2) Set up outer (F-score) and inner (hyperparam) CVs
3) Loop through labels
4) Loop through outer folds
5) Perform grid search to determine best hyperparams with inner CV
6) Use best hyperparams from inner CV to train/test that fold in outer CV
7) Save hyperparams used in outer CV folds (10) and trials (30) to file (n_rows=300)
  - Variability of hyperparams will be evaluated.
8) Save predictions (across folds?) for each trial, to be evaluated by Cody.
  - The predictions from each test fold in the outer CV will be combined in
    one df. One df will be made for each trial (n=30). This means there will be
    30 sets of predictions that will need to be combined in some way (majority
    vote?) in order to be evaluated.

NOTE: This doesn't take dimension into account at all. In order to employ
      transmogrification (dimension-wise feature boosting), we'll need to split
      labels by dimension and feed predictions from each dimension into the
      features for the next.
"""

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from os.path import join


def run_svm_cogat_cv(label_df, features_df, out_dir):
    ## Settings
    space = 'cogat'
    classifier = 'svm'
    source = 'abstract'

    # We will use a Support Vector Classifier with "rbf" kernel
    svm = SVC(kernel='rbf', class_weight='balanced')

    # Set up possible values of parameters to optimize over
    p_grid = {'C': [1, 10, 100],
              'gamma': [.01, .1, 1.]}

    # Get data from DataFrame
    features = features_df.as_matrix()

    # Pull info from label_df
    labels = label_df.as_matrix()[:, :6]
    label_names = label_df.columns.tolist()[:6]

    # Loop for each trial
    rows = []

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset. Classic 5x2 split.
    # 5x2 popularized in 
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3325&rep=rep1&type=pdf
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    preds_array = np.zeros(labels.shape)
    f_alllabels = []
    for i_label in range(labels.shape[1]):
        test_label = label_names[i_label]
        print('{0}'.format(test_label))
        f_label_row = []
        X_range = np.zeros((labels.shape[0], 1))
        for j_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_range,
                                                                      labels[:, i_label])):
            print('\t{0}'.format(j_fold))

            # Define gaz, extract features, and perform feature selection here.
            tfidf = TfidfTransformer()

            # Get classes.
            y_train = labels[train_idx, i_label]
            y_test = labels[test_idx, i_label]

            # Get raw data.
            tfidf.fit(features[train_idx, :])
            transformed_features = tfidf.transform(features)

            # Do we choose best params across labels or per label?
            # Let's say per label for now, so Cody can analyze variability
            # between labels/dimensions as well as between folds.
            X_train = transformed_features[train_idx, :]
            X_test = transformed_features[test_idx, :]

            # Select hyperparameters using inner CV.
            gs_clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
            gs_clf.fit(X_train, y_train)

            # Train/test outer CV using best parameters from inner CV.
            params = gs_clf.best_params_

            # Track best parameters from inner cvs and use to train outer CV model.
            cv_clf = svm.set_params(C=params['C'], gamma=params['gamma'])
            cv_clf.fit(X_train, y_train)
            preds = cv_clf.predict(X_test)

            f_fold_label = f1_score(y_test, preds)
            f_label_row.append(f_fold_label)

            # Write out 1x[nTest] array of predictions to file.
            filename = 'preds_{0}_{1}_{2}_{3}_{4}.csv'.format(classifier,
                                                              source, space,
                                                              i_label, j_fold)
            dat = np.vstack((preds, y_test)).transpose()
            df = pd.DataFrame(data=dat,
                              columns=['predictions', 'true'])
            df.to_csv(join(out_dir, filename), index=False)

            # Add new predictions to overall array.
            preds_array[test_idx, i_label] = preds

            # Put hyperparameters in dataframe.
            row = [label_names[i_label], classifier, source, space,
                   j_fold, params['C'], params['gamma']]
            rows += [row]
        f_alllabels += [f_label_row]

    # Write out [nFolds]x[nLabels] array of F1-scores to file.
    f_filename = '{0}_{1}_{2}_f1.csv'.format(classifier, source, space)
    f_cols = ['Fold_{0}'.format(f) for f in range(j_fold+1)]
    print f_alllabels
    df = pd.DataFrame(data=f_alllabels, columns=f_cols, index=label_names)
    df.index.name = 'label'
    df.to_csv(join(out_dir, f_filename))

    # Save predictions array to file.
    p_filename = '{0}_{1}_{2}_preds.csv'.format(classifier, source, space)
    df = pd.DataFrame(data=preds_array, columns=label_names, index=label_df.index)
    df.index.name = 'pmid'
    df.to_csv(join(out_dir, p_filename))

    # Save hyperparameter values to file.
    hp_filename = '{0}_{1}_{2}_params.csv'.format(classifier, source, space)
    hp_cols = ['label', 'classifier', 'feature source', 'feature space',
               'fold', 'C', 'gamma']
    df = pd.DataFrame(data=rows, columns=hp_cols)
    df.to_csv(join(out_dir, hp_filename), index=False)


#def run():
data_dir = '/home/data/nbc/athena/athena-data2/'
label_df = pd.read_csv(join(data_dir, 'labels/red_labels.csv'),
                       index_col='pmid')
features_df = pd.read_csv(join(data_dir, 'features/cogat_abstract.csv'),
                          index_col='pmid')
label_pmids = set(label_df.index.values)
feature_pmids = set(features_df.index.values)
shared_pmids = sorted(list(label_pmids & feature_pmids))
label_df = label_df.loc[shared_pmids]
features_df = features_df.loc[shared_pmids]

out_dir = '/scratch/tsalo006/test_cv/'
run_svm_cogat_cv(label_df, features_df, out_dir)
    