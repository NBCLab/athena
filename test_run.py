import data_preparation
import run_cv
import pandas as import pd
from os import join

data_dir = '/home/data/nbc/athena/athena-data/'
label_file = '/home/data/nbc/athena/athena-data/labels/full.csv'

data_preparation.process_raw_data(data_dir)


label_df = pd.read_csv(label_file, index_col='pmid')
pmids = label_df.index.tolist()

# SVMs
for source in ['abstract', 'full']:
    # Bag of words
    text_dir = join(data_dir, 'text/stemmed_{0}'.format(source))
    data = []
    for pmid in pmids:
        with open(join(text_dir, pmid+'.txt'), 'rb') as fo:
            text = fo.read()
            data.append([[pmid, text]])
    text_df = pd.DataFrame(columns=['pmid', 'text'], data=data)
    run_svm_bow_cv(label_df, text_df, source)

    # CogAt
    text_dir = join(data_dir, 'text/processed_{0}'.format(source))
    data = []
    for pmid in pmids:
        with open(join(text_dir, pmid+'.txt'), 'rb') as fo:
            text = fo.read()
            data.append([[pmid, text]])
    text_df = pd.DataFrame(columns=['pmid', 'text'], data=data)
    run_svm_cogat_cv(label_df, text_df, source)

# KNN
for source in ['abstract', 'full']:
    # Bag of words
    text_dir = join(data_dir, 'text/stemmed_{0}'.format(source))
    data = []
    for pmid in pmids:
        with open(join(text_dir, pmid+'.txt'), 'rb') as fo:
            text = fo.read()
            data.append([[pmid, text]])
    text_df = pd.DataFrame(columns=['pmid', 'text'], data=data)
    run_knn_bow_cv(label_df, text_df, source)

    # CogAt
    text_dir = join(data_dir, 'text/processed_{0}'.format(source))
    data = []
    for pmid in pmids:
        with open(join(text_dir, pmid+'.txt'), 'rb') as fo:
            text = fo.read()
            data.append([[pmid, text]])
    text_df = pd.DataFrame(columns=['pmid', 'text'], data=data)
    run_knn_cogat_cv(label_df, text_df, source)

# Naive Bayes
for source in ['abstract', 'full']:
    # Bag of words
    text_dir = join(data_dir, 'text/stemmed_{0}'.format(source))
    data = []
    for pmid in pmids:
        with open(join(text_dir, pmid+'.txt'), 'rb') as fo:
            text = fo.read()
            data.append([[pmid, text]])
    text_df = pd.DataFrame(columns=['pmid', 'text'], data=data)
    run_nbayes_bow_cv(label_df, text_df, source)

    # CogAt
    text_dir = join(data_dir, 'text/processed_{0}'.format(source))
    data = []
    for pmid in pmids:
        with open(join(text_dir, pmid+'.txt'), 'rb') as fo:
            text = fo.read()
            data.append([[pmid, text]])
    text_df = pd.DataFrame(columns=['pmid', 'text'], data=data)
    run_nbayes_cogat_cv(label_df, text_df, source)

# Logistic Regression
for source in ['abstract', 'full']:
    # Bag of words
    text_dir = join(data_dir, 'text/stemmed_{0}'.format(source))
    data = []
    for pmid in pmids:
        with open(join(text_dir, pmid+'.txt'), 'rb') as fo:
            text = fo.read()
            data.append([[pmid, text]])
    text_df = pd.DataFrame(columns=['pmid', 'text'], data=data)
    run_lreg_bow_cv(label_df, text_df, source)

    # CogAt
    text_dir = join(data_dir, 'text/processed_{0}'.format(source))
    data = []
    for pmid in pmids:
        with open(join(text_dir, pmid+'.txt'), 'rb') as fo:
            text = fo.read()
            data.append([[pmid, text]])
    text_df = pd.DataFrame(columns=['pmid', 'text'], data=data)
    run_lreg_cogat_cv(label_df, text_df, source)
