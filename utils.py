# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:52:44 2016

@author: salo
"""

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd

stemmer = PorterStemmer()
stop = stopwords.words("english")


def stem_tokens(tokens, stemmer):
    """
    http://stackoverflow.com/questions/26126442/combining-text-stemming-and-
    removal-of-punctuation-in-nltk-and-scikit-learn
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    """
    http://stackoverflow.com/questions/26126442/combining-text-stemming-and-
    removal-of-punctuation-in-nltk-and-scikit-learn
    """
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def cogpo_columns(columns):
    """
    """
    column_to_cogpo = {"Paradigm Class": "ParadigmClass",
                       "Behavioral Domain": "BehavioralDomain",
                       "Diagnosis": "Diagnosis",
                       "Stimulus Modality": "StimulusModality",
                       "Stimulus Type": "StimulusType",
                       "Response Modality": "OvertResponseModality",
                       "Response Type": "OvertResponseType",
                       "Instructions": "Instruction",
                       "Context": "Context"}
    subset = { key:value for key, value in column_to_cogpo.items() if key in columns }
    return subset


def clean_str(str_):
    """
    """
    label = str_.replace(' ', '').replace("'", '').replace('(', '.').replace(')', '').replace('Stroop-', 'Stroop.')
    return label


def get_label_parents(df, column, dimension):
    """
    Create full list of labels (and their parents) in DataFrame column.
    """
    col_labels = df[pd.notnull(df[column])][column]
    col_labels.apply(lambda x: '{%s}' % '| '.join(x))
    col_labels = col_labels.tolist()
    dim_labels = [clean_str(label) for exp_labels in col_labels for label in exp_labels.split('| ')]

    parents = dim_labels[:]
    while parents:
        parents = ['.'.join(item.split('.')[:-1]) for item in parents if len(item.split('.'))>1]
        dim_labels += parents
    dim_labels = ['{0}.{1}'.format(dimension, label) for label in dim_labels]
    return table
