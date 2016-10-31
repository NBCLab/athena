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
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import BernoulliNB as BNB
from sklearn.linear_model import LogisticRegression as LR

from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

stop = stopwords.words("english")


def run_svm_bow_cv(label_df, text_df, source):
    ## Settings
    space = 'bow'
    classifier = 'svm'

    # We will use a Support Vector Classifier with "rbf" kernel
    svm = SVC(kernel='rbf', class_weight='balanced')

    # Set up possible values of parameters to optimize over
    p_grid = {"C": [1, 10, 100],
              "gamma": [.01, .1, 1.]}

    # Get data from DataFrame
    texts = text_df['text'].tolist()

    # Pull info from label_df
    labels = label_df.as_matrix()
    label_names = label_df.columns.tolist()

    # Loop for each trial
    rows = []

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset. Classic 5x2 split.
    # 5x2 popularized in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3325&rep=rep1&type=pdf
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    preds_array = np.zeros(labels.shape)
    f_alllabels = []
    for i_label in range(labels.shape[1]):
        test_label = label_names[i_label]
        print('{0}'.format(test_label))
        f_label_row = []
        for j_fold, (train_idx, test_idx) in enumerate(outer_cv.split(features,
                                                                      labels[:, i_label])):
            print('\t{0}'.format(j_fold))

            # Define gaz, extract features, and perform feature selection here.
            tfidf = TfidfVectorizer(stop_words=stop,
                            token_pattern="(?!\\[)[A-z\\-]{3,}",
                            ngram_range=(1, 2),
                            sublinear_tf=True,
                            min_df=75, max_df=0.8, max_features=2000)

            # TODO: Feature selection.
            print('\t\tDo feature selection.')

            # Get features.
            train_texts = [texts[i] for i in train_idx]
            tfidf.fit(train_texts)
            features = tfidf.transform(texts)

            # Do we choose best params across labels or per label?
            # Let's say per label for now, so Cody can analyze variability
            # between labels/dimensions as well as between folds.
            X_train = features[train_idx, :]
            y_train = labels[train_idx, i_label]
            X_test = features[test_idx, :]
            y_test = labels[test_idx, i_label]

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
            f_label_row += f_fold_label

            # Write out 1x[nTest] array of predictions to file.
            filename = 'preds_{0}_{1}_{2}_{3}_{4}.csv'.format(classifier,
                                                              source, space,
                                                              i_label, j_fold)
            df = pd.DataFrame(data=[preds, y_test],
                              columns=['predictions', 'true'])
            df.to_csv(filename, index=False)

            # Add new predictions to overall array.
            preds_array[test_idx, i_label] = preds

            # Put hyperparameters in dataframe.
            row = [label_names[i_label], classifier, source, space,
                   j_fold, params['C'], params['gamma']]
            rows += [row]
        f_alllabels += [f_label_row]

    # Write out [nFolds]x[nLabels] array of F1-scores to file.
    f_filename = '{0}_{1}_{2}_f1.csv'.format(classifier, source, space)
    f_cols = ['Fold_{0}'.format(f) for f in range(j_fold)]
    df = pd.DataFrame(data=f_alllabels, columns=f_cols, index=label_names)
    df.index.name = 'label'
    df.to_csv(f_filename)

    # Save predictions array to file.
    p_filename = '{0}_{1}_{2}_preds.csv'.format(classifier, source, space)
    df = pd.DataFrame(data=preds_array, columns=label_names, index=label_df.index)
    df.index.name = 'pmid'
    df.to_csv(p_filename)

    # Save hyperparameter values to file.
    hp_filename = '{0}_{1}_{2}_params.csv'.format(classifier, source, space)
    hp_cols = ['label', 'classifier', 'feature source', 'feature space',
               'fold', 'C', 'gamma']
    df = pd.DataFrame(data=rows, columns=hp_cols)
    df.to_csv(hp_filename, index=False)
