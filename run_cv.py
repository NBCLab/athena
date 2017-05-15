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
import numpy as np
import pandas as pd
from glob import glob
import multiprocessing as mp
from os.path import basename, splitext, join

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

stop = stopwords.words('english')


def run_bow_cv(label_df, text_dir, out_dir, classifier, source):
    ## Settings
    space = 'bow'
    n_cogat = 1754
    n_iters = 5
    
    if classifier == 'svm':
        # We will use a Support Vector Classifier with "rbf" kernel
        clf = SVC(kernel='rbf', class_weight='balanced')
    
        # Set up possible values of parameters to optimize over
        p_grid = {'C': [1, 10, 100],
                  'gamma': [.01, .1, 1.]}
    elif classifier == 'bnb':
        clf = BernoulliNB(fit_prior=True)

        # Set up possible values of parameters to optimize over
        p_grid = {'alpha': [0.01, 0.1, 1, 10]}
    elif classifier == 'lr':
        clf = LogisticRegression(class_weight='balanced')
        
        # Set up possible values of parameters to optimize over
        p_grid = {'C': [.01, .1, 1, 10, 100],
                  'penalty': ['l1', 'l2']}
    elif classifier == 'knn':
        clf = KNeighborsClassifier()
        
        # Set up possible values of parameters to optimize over
        p_grid = {'n_neighbors': [1, 3, 5, 7, 9],
                  'p': [1, 2],
                  'weights': ['uniform', 'distance']}
    else:
        raise Exception('Classifier {0} not supported.'.format(classifier))
        
    param_cols = sorted(p_grid.keys())
    
    # Get data from DataFrame
    pmids = label_df.index.values
    texts = []
    for pmid in pmids:
        with open(join(text_dir, '{0}.txt'.format(pmid)), 'r') as fo:
            texts.append(fo.read())
    feature_range = np.arange(len(texts))

    # Pull info from label_df
    labels = label_df.as_matrix()[:, :6]
    label_names = label_df.columns.tolist()[:6]

    f_alllabels = []  # One F1 array for all iters/folds/labels
    for iter_ in range(n_iters):
        print('{0}'.format(iter_))

        # Loop for each trial
        sel_params = []  # One param array for each iter
        preds_array = np.zeros(labels.shape)  # One pred array for each iter

        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset. Classic 5x2 split.
        # 5x2 popularized in 
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3325&rep=rep1&type=pdf
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=iter_)
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=iter_)

        for i_label in range(labels.shape[1]):
            test_label = label_names[i_label]
            print('\t{0}'.format(test_label))
            f_label_row = []
            for j_fold, (train_idx, test_idx) in enumerate(outer_cv.split(feature_range,
                                                                          labels[:, i_label])):
                print('\t\t{0}'.format(j_fold))

                # Define gaz, extract features, and perform feature selection here.
                tfidf = TfidfVectorizer(stop_words=stop,
                                        token_pattern='(?!\\[)[A-z\\-]{3,}',
                                        ngram_range=(1, 2),
                                        sublinear_tf=True,
                                        min_df=80, max_df=1.)

                # Get classes.
                y_train = labels[train_idx, i_label]
                y_test = labels[test_idx, i_label]

                # Get raw data.
                train_texts = [texts[i] for i in train_idx]

                # Feature selection.
                features = tfidf.fit_transform(train_texts)
                names = tfidf.get_feature_names()
                
                # Perform feature selection if there are too many features.
                if len(names) > n_cogat:
                    skb = SelectKBest(chi2, k=n_cogat)
                    skb.fit(features, y_train)
                    neg_n = -1 * n_cogat
                    keep_idx = np.argpartition(skb.scores_, neg_n)[neg_n:]
                    vocabulary = [str(names[i]) for i in keep_idx]
                else:
                    vocabulary = names[:]
                
                # We probably want to store the top words for each fold/label to
                # measure stability or something.
                vocab_filename = '{c}_{so}_{sp}_{l}_i{i}_f{f}_feats.csv'.format(c=classifier,
                                                                                so=source,
                                                                                sp=space,
                                                                                l=test_label,
                                                                                i=iter_,
                                                                                f=j_fold)
                vocab_df = pd.DataFrame(data=vocabulary, columns=['features'])
                vocab_df.to_csv(join(out_dir, vocab_filename), index=False)

                # Now feature extraction with the new vocabulary
                tfidf = TfidfVectorizer(stop_words=stop,
                                        vocabulary=vocabulary,
                                        token_pattern='(?!\\[)[A-z\\-]{3,}',
                                        ngram_range=(1, 2),
                                        sublinear_tf=True,
                                        min_df=80, max_df=1.)
                tfidf.fit(train_texts)
                features = tfidf.transform(texts)

                # Do we choose best params across labels or per label?
                # Let's say per label for now, so Cody can analyze variability
                # between labels/dimensions as well as between folds.
                X_train = features[train_idx, :]
                X_test = features[test_idx, :]

                # Select hyperparameters using inner CV.
                gs_clf = GridSearchCV(estimator=clf, param_grid=p_grid, cv=inner_cv)
                gs_clf.fit(X_train, y_train)

                # Train/test outer CV using best parameters from inner CV.
                params = gs_clf.best_params_

                # Track best parameters from inner cvs and use to train outer CV model.
                if classifier == 'svm':
                    cv_clf = clf.set_params(C=params['C'], gamma=params['gamma'])
                elif classifier == 'bnb':
                    cv_clf = clf.set_params(alpha=params['alpha'])
                elif classifier == 'lr':
                    cv_clf = clf.set_params(C=params['C'], penalty=params['penalty'])
                elif classifier == 'knn':
                    cv_clf = clf.set_params(n_neighbors=params['n_neighbors'],
                                            p=params['p'], weights=params['weights'])
                
                cv_clf.fit(X_train, y_train)
                preds = cv_clf.predict(X_test)

                f_fold_label = f1_score(y_test, preds)
                f_label_row = [classifier, source, space, j_fold, test_label,
                               iter_, f_fold_label]
                f_alllabels += [f_label_row]
                
                # Add new predictions to overall array.
                preds_array[test_idx, i_label] = preds

                # Put hyperparameters in dataframe.
                p_vals = [params[key] for key in param_cols]
                p_row = [classifier, source, space, label_names[i_label],
                         j_fold, iter_] + p_vals
                sel_params += [p_row]

        # Save predictions array to file.
        p_filename = '{0}_{1}_{2}_{3}_preds.csv'.format(classifier, source, space, iter_)
        df = pd.DataFrame(data=preds_array, columns=label_names, index=label_df.index)
        df.index.name = 'pmid'
        df.to_csv(join(out_dir, p_filename))

        # Save hyperparameter values to file.
        hp_filename = '{0}_{1}_{2}_{3}_params.csv'.format(classifier, source, space, iter_)
        hp_cols = ['classifier', 'source', 'space', 'label', 'fold', 'iter'] + param_cols
        df = pd.DataFrame(data=sel_params, columns=hp_cols)
        df.to_csv(join(out_dir, hp_filename), index=False)

    # Write out [nFolds*nIters]x[nLabels] array of F1-scores to file.
    f_filename = '{0}_{1}_{2}_f1.csv'.format(classifier, source, space)
    f_cols = ['classifier', 'source', 'space', 'fold', 'label', 'iter', 'f1']
    
    df = pd.DataFrame(data=f_alllabels, columns=f_cols)
    df.to_csv(join(out_dir, f_filename), index=False)


def run_cogat_cv(label_df, features_df, out_dir, classifier, source):
    ## Settings
    space = 'cogat'
    n_iters = 5

    if classifier == 'svm':
        # We will use a Support Vector Classifier with "rbf" kernel
        clf = SVC(kernel='rbf', class_weight='balanced')
    
        # Set up possible values of parameters to optimize over
        p_grid = {'C': [1, 10, 100],
                  'gamma': [.01, .1, 1.]}
    elif classifier == 'bnb':
        clf = BernoulliNB(fit_prior=True)

        # Set up possible values of parameters to optimize over
        p_grid = {'alpha': [0.01, 0.1, 1, 10]}
    elif classifier == 'lr':
        clf = LogisticRegression(class_weight='balanced')
        
        # Set up possible values of parameters to optimize over
        p_grid = {'C': [.01, .1, 1, 10, 100],
                  'penalty': ['l1', 'l2']}
    elif classifier == 'knn':
        clf = KNeighborsClassifier()
        
        # Set up possible values of parameters to optimize over
        p_grid = {'n_neighbors': [1, 3, 5, 7, 9],
                  'p': [1, 2],
                  'weights': ['uniform', 'distance']}
    else:
        raise Exception('Classifier {0} not supported.'.format(classifier))
        
    param_cols = sorted(p_grid.keys())

    # Get data from DataFrame
    features = features_df.as_matrix()

    # Pull info from label_df
    labels = label_df.as_matrix()[:, :6]
    label_names = label_df.columns.tolist()[:6]

    f_alllabels = []  # One F1 array for all iters/folds/labels
    for iter_ in range(n_iters):
        print('{0}'.format(iter_))

        # Loop for each trial
        sel_params = []  # One param array for each iter
        preds_array = np.zeros(labels.shape)  # One pred array for each iter

        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset. Classic 5x2 split.
        # 5x2 popularized in 
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3325&rep=rep1&type=pdf
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=iter_)
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=iter_)

        for i_label in range(labels.shape[1]):
            test_label = label_names[i_label]
            print('\t{0}'.format(test_label))

            X_range = np.zeros((labels.shape[0], 1))
            for j_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_range,
                                                                          labels[:, i_label])):
                print('\t\t{0}'.format(j_fold))

                # Define gaz, extract features, and perform feature selection here.
                tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True,
                                         sublinear_tf=True)

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
                gs_clf = GridSearchCV(estimator=clf, param_grid=p_grid, cv=inner_cv)
                gs_clf.fit(X_train, y_train)

                # Train/test outer CV using best parameters from inner CV.
                params = gs_clf.best_params_

                # Track best parameters from inner cvs and use to train outer CV model.
                if classifier == 'svm':
                    cv_clf = clf.set_params(C=params['C'], gamma=params['gamma'])
                elif classifier == 'bnb':
                    cv_clf = clf.set_params(alpha=params['alpha'])
                elif classifier == 'lr':
                    cv_clf = clf.set_params(C=params['C'], penalty=params['penalty'])
                elif classifier == 'knn':
                    cv_clf = clf.set_params(n_neighbors=params['n_neighbors'],
                                            p=params['p'], weights=params['weights'])
                
                cv_clf.fit(X_train, y_train)
                preds = cv_clf.predict(X_test)

                f_fold_label = f1_score(y_test, preds)
                f_label_row = [classifier, source, space, j_fold, test_label,
                               iter_, f_fold_label]
                f_alllabels += [f_label_row]
                
                # Add new predictions to overall array.
                preds_array[test_idx, i_label] = preds

                # Put hyperparameters in dataframe.
                p_vals = [params[key] for key in param_cols]
                p_row = [classifier, source, space, label_names[i_label],
                         j_fold, iter_] + p_vals
                sel_params += [p_row]

        # Save predictions array to file.
        p_filename = '{0}_{1}_{2}_{3}_preds.csv'.format(classifier, source, space, iter_)
        df = pd.DataFrame(data=preds_array, columns=label_names, index=label_df.index)
        df.index.name = 'pmid'
        df.to_csv(join(out_dir, p_filename))

        # Save hyperparameter values to file.
        hp_filename = '{0}_{1}_{2}_{3}_params.csv'.format(classifier, source, space, iter_)
        hp_cols = ['classifier', 'source', 'space', 'label', 'fold', 'iter'] + param_cols
        df = pd.DataFrame(data=sel_params, columns=hp_cols)
        df.to_csv(join(out_dir, hp_filename), index=False)

    # Write out [nFolds*nIters]x[nLabels] array of F1-scores to file.
    f_filename = '{0}_{1}_{2}_f1.csv'.format(classifier, source, space)
    f_cols = ['classifier', 'source', 'space', 'fold', 'label', 'iter', 'f1']
    
    df = pd.DataFrame(data=f_alllabels, columns=f_cols)
    df.to_csv(join(out_dir, f_filename), index=False)


def run(data_dir, out_dir):
    label_df = pd.read_csv(join(data_dir, 'labels/red_labels.csv'),
                           index_col='pmid')

    sources = ['full', 'abstract']
    classifiers = ['knn', 'svm', 'bnb', 'lr']
    
    # BOW PMIDs
    text_dirs = [join(data_dir, 'text/stemmed_{0}/'.format(s)) for s in sources]
    texts = [glob(join(td, '*.txt')) for td in text_dirs]
    bow_pmids_ = [[int(splitext(basename(f))[0]) for f in t] for t in texts]
    bow_pmids = sorted(list(set(bow_pmids_[0]).intersection(bow_pmids_[1])))
    
    # CogAt PMIDs
    cogat_dfs = [pd.read_csv(join(data_dir, 'features/cogat_{0}.csv'.format(s)),
                                  index_col='pmid') for s in sources]
    cogat_pmids_ = [df.index.tolist() for df in cogat_dfs]
    cogat_pmids = sorted(list(set(cogat_pmids_[0]).intersection(cogat_pmids_[1])))
    
    # Label PMIDs
    label_pmids = label_df.index.tolist()
    
    # Intersection between all three sets
    shared_pmids = set(label_pmids).intersection(bow_pmids).intersection(cogat_pmids)
    shared_pmids = sorted(list(shared_pmids))
    
    # Reduce corpus by PMIDs with features and labels
    label_df = label_df.loc[shared_pmids]
    cogat_dfs = [df.loc[shared_pmids] for df in cogat_dfs]
    
    for i, s in enumerate(sources):
        for c in classifiers:
            # BOW
            run_bow_cv(label_df, text_dirs[i], out_dir, source=s, classifier=c)

            # CogAt
            run_cogat_cv(label_df, cogat_dfs[i], out_dir, source=s, classifier=c)


def run_para(data_dir, out_dir):
    sources = ['full', 'abstract']
    classifiers = ['knn', 'svm', 'bnb', 'lr']
    
    label_df = pd.read_csv(join(data_dir, 'labels/red_labels.csv'),
                           index_col='pmid')

    # BOW PMIDs
    text_dirs = [join(data_dir, 'text/stemmed_{0}/'.format(s)) for s in sources]
    texts = [glob(join(td, '*.txt')) for td in text_dirs]
    bow_pmids_ = [[int(splitext(basename(f))[0]) for f in t] for t in texts]
    bow_pmids = sorted(list(set(bow_pmids_[0]).intersection(bow_pmids_[1])))
    
    # CogAt PMIDs
    cogat_dfs = [pd.read_csv(join(data_dir, 'features/cogat_{0}.csv'.format(s)),
                                  index_col='pmid') for s in sources]
    cogat_pmids_ = [df.index.tolist() for df in cogat_dfs]
    cogat_pmids = sorted(list(set(cogat_pmids_[0]).intersection(cogat_pmids_[1])))
    
    # Label PMIDs
    label_pmids = label_df.index.tolist()
    
    # Intersection between all three sets
    shared_pmids = set(label_pmids).intersection(bow_pmids).intersection(cogat_pmids)
    shared_pmids = sorted(list(shared_pmids))
    
    # Reduce corpus by PMIDs with features and labels
    label_df = label_df.loc[shared_pmids]
    
    label_dfs = []
    out_dirs = []
    text_dirs = []
    cogat_dfs = []
    so_input = []
    clf_input = []

    for s in sources:
        text_dir = join(data_dir, 'text/stemmed_{0}/'.format(s))
        cogat_df = pd.read_csv(join(data_dir, 'features/cogat_{0}.csv'.format(s)),
                               index_col='pmid')
        cogat_df = cogat_df.loc[shared_pmids]
        for c in classifiers:
            label_dfs.append(label_df)
            out_dirs.append(out_dir)
            text_dirs.append(text_dir)
            cogat_dfs.append(cogat_df)
            so_input.append(s)
            clf_input.append(c)
    params = zip(*[label_dfs, out_dirs, text_dirs,
                   cogat_dfs, so_input, clf_input])
    pool = mp.Pool(len(params))
    pool.map(_run, params)
    pool.close()


def _run(params):
    label_df, out_dir, text_dir, cogat_df, s, c = params
    
    # BOW
    run_bow_cv(label_df, text_dir, out_dir, source=s, classifier=c)
    
    # CogAt
    run_cogat_cv(label_df, cogat_df, out_dir, source=s, classifier=c)


if __name__ == '__main__':
    data_dir = '/home/data/nbc/athena/athena-data2/'
    out_dir = '/scratch/tsalo006/test_cv2/'
    run_para(data_dir, out_dir)

