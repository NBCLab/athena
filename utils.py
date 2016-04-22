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
    column_to_cogpo = {"Paradigm Class": "Experiments.ParadigmClass",
                       "Behavioral Domain": "Experiments.BehavioralDomain",
                       "Diagnosis": "Subjects.Diagnosis",
                       "Stimulus Modality": "Conditions.StimulusModality",
                       "Stimulus Type": "Conditions.StimulusType",
                       "Response Modality": "Conditions.OvertResponseModality",
                       "Response Type": "Conditions.OvertResponseType",
                       "Instructions": "Conditions.Instruction"}
    subset = { key:value for key, value in column_to_cogpo.items() if key in columns }
    return subset


def clean_str(str_):
    label = str_.replace(" ", "").replace("'", "").replace("(", ".").replace(")", "").replace("Stroop-", "Stroop.")
    return label


def df_to_list(df, column_name, prefix):
    table = df[pd.notnull(df[column_name])][column_name]
    table.apply(lambda x: "{%s}" % "| ".join(x))
    table = table.tolist()
    table = [clean_str(item) for sublist in table for item in sublist.split("| ")]
    
    parents = table
    while parents:
        parents = [".".join(item.split(".")[:-1]) for item in parents if len(item.split("."))>1]
        table += parents
    table = ["{0}.{1}".format(prefix, item) for item in table]
    return table
