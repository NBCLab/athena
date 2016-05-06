# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:50:38 2016

Data processing functions.

@author: salo
"""

import os
from glob import glob
import pandas as pd
import random
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import PorterStemmer
from utils import cogpo_columns, clean_str, df_to_list


def stem_corpus(data_dir="/home/data/nbc/athena/v1.1-data/"):
    for text_type in ["combined", "full"]:
        print("Processing {0}".format(text_type))
        text_dir = os.path.join(data_dir, "text/", text_type)
        stem_dir = os.path.join(data_dir, "text/", "stemmed_"+text_type)
    
        stemmer = EnglishStemmer()
        stemmerTest = PorterStemmer()
        
        for file_ in glob(os.path.join(text_dir, "*.txt")):
            filename = os.path.basename(file_)
            print("\tStemming {0}".format(filename))
            with open(file_, "r") as fo:
                text = fo.read()
        
            stem_list = []

            for word in text.split():
                try:
                    test = " ".join(["kdkd", stemmerTest.stem(word), "kdkd"])
                    
                except:
                    word = word.decode('utf8', 'ignore').encode('ascii','ignore')
                    
                stem_list.append(stemmer.stem(word))
        
            with open(os.path.join(stem_dir, filename), "w") as fo:
                fo.write(" ".join(stem_list))



def label_data(data_dir="/home/data/nbc/athena/v1.1-data/"):
    """
    Convert metadata files to instance-by-label matrix.
    """
    metadata_dir = os.path.join(data_dir, "metadata/")
    filenames = sorted(glob(os.path.join(metadata_dir, "*.csv")))
    
    columns = ["Paradigm Class", "Behavioral Domain", "Diagnosis",
               "Stimulus Modality", "Stimulus Type", "Response Modality",
               "Response Type", "Instructions", "Context"]
    column_to_cogpo = cogpo_columns(columns)
    
    full_cogpo = []
    metadata_dfs = [pd.read_csv(file_, dtype=str)[["PubMed ID"] + column_to_cogpo.keys()] for file_ in filenames]
    metadata_df = pd.concat(metadata_dfs, ignore_index=True)
    metadata_df = metadata_df[metadata_df["PubMed ID"].str.contains("^\d+$")].reset_index()
    list_of_metadata_pmids = metadata_df["PubMed ID"].unique().tolist()
    
    for column in column_to_cogpo.keys():
        table = df_to_list(metadata_df, column, column_to_cogpo[column])
        full_cogpo += table
    
    full_cogpo = sorted(list(set(full_cogpo)))
    
    for text_type in ["full", "combined"]:
        type_dir = os.path.join(data_dir, text_type)
        text_dir = os.path.join(data_dir, "text/", text_type)

        # Preallocate label DataFrame
        list_of_files = [os.path.splitext(file_)[0] for file_ in os.listdir(text_dir)]
        list_of_files = sorted(list(set(list_of_files)))
        list_of_pmids = sorted(list(set(list_of_metadata_pmids).intersection(list_of_files)))
        
        column_names = ["pmid"] + full_cogpo
        label_df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
        label_df["pmid"] = list_of_pmids
        
        for row in metadata_df.index:
            pmid = metadata_df["PubMed ID"].iloc[row]
            if pmid in list_of_pmids:
                for column in column_to_cogpo.keys():
                    values = metadata_df[column].iloc[row]
                    if pd.notnull(values):
                        values = values.split("| ")
                        values = ["{0}.{1}".format(column_to_cogpo[column], clean_str(item)) for item in values]
                        for value in values:
                            for out_column in label_df.columns:
                                if out_column in value:
                                    ind = label_df.loc[label_df["pmid"]==pmid].index[0]
                                    label_df[out_column].iloc[ind] = 1
        
        # Reduce DataFrame.
        # Only include labels with at least 30 samples
        min_ = 30
        label_counts = label_df.sum()
        keep_labels = label_counts[label_counts>=min_].index
        label_df = label_df[keep_labels]
        label_df = label_df[(label_df.T != 0).any()]
        count_df = label_df.sum().to_frame()
        label_df = label_df.astype(int).astype(str)
        out_file = os.path.join(type_dir, "labels/full.csv")
        label_df.to_csv(out_file, index=False)
    
        count_file = os.path.join(type_dir, "labels/labels.csv")
        count_df = count_df.ix[1:]
        count_df.index.name = "Label"
        count_df.columns = ["Count"]
        count_df.to_csv(count_file, index=True)


def split_data(labels_file, test_percent=0.3):
    """
    Simply perform train/test data split.
    """
    data_dir = os.path.dirname(labels_file)
    
    all_labels = pd.read_csv(labels_file)
    
    column_names = all_labels.columns.values
    all_data = all_labels.as_matrix()
    de_ided = all_data[:, 1:]
    
    n_instances = all_data.shape[0]
    index = range(n_instances)
    
    n_test = int(n_instances * test_percent)
    n_train = n_instances - n_test
    print("Size of test dataset: {0}".format(n_test))
    print("Size of training dataset: {0}".format(n_train))
    
    shuf_index = index[:]
    random.shuffle(shuf_index)
    test_rows = de_ided[shuf_index[:n_test]]
    train_rows = de_ided[shuf_index[n_test:]]
    
    train_data = all_data[sorted(shuf_index[n_test:]), :]
    test_data = all_data[sorted(shuf_index[:n_test]), :]
    df_train = pd.DataFrame(columns=column_names, data=train_data)
    df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    df_test = pd.DataFrame(columns=column_names, data=test_data)
    df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
