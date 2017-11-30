# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 00:07:14 2016

Prepare data for extraction and analysis.

Inputs:
- IDs
- Raw text
- Raw labels (metadata)

Outputs:
- Processed text
- Labeled data
- Gazetteers

@author: salo
"""

"""
First round of commenting: Tue Oct 4 2016 Cody Riedel
"""

import os
from os.path import join
import re
import csv
from glob import glob

import numpy as np
import pandas as pd
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import PorterStemmer

import extract_cogat as ec
import abbr

from utils import clean_str, get_label_parents

with open('/home/data/nbc/athena-data/misc/label_converter.csv', mode='r') as infile:
    reader = csv.reader(infile)
    label_con = {row[0]:row[1] for row in reader}


def process_raw_data(data_dir='/home/data/nbc/athena/athena-data/'):
    """ Defines function process_raw_data which works on the athena data directory
    Label data using metadata files, split data into training and test
    datasets.
    """

    #   Calls the process_corpus function, defined below
    #   process_corpus reads in the text, performs abbreviation, spelling,
    # translation, and overall text Processing
    #   process_corpus outputs the processed text for each file and the stemmed file
    for feature_source in ['abstract', 'full']:
        process_corpus(data_dir, feature_source)

    #   Calls the label_data function, defined below
    #   label_data reads in the metadata csv files, concatenates them, then
    # reads in the processed text files
    #   label_data outputs a binary pmid by label metadata matrix
    label_data(data_dir)
    generate_gazetteer(data_dir)


def process_corpus(data_dir, feature_source):
    """ Performs all the data pre-processing, reading in text, removing some
    abbreviations, text removal, writing out stem files and processed text.
    """
    raw_dir = os.path.join(data_dir, 'text/', feature_source)
    stem_dir = os.path.join(data_dir, 'text/stemmed_{0}/'.format(feature_source))
    proc_dir = os.path.join(data_dir, 'text/processed_{0}/'.format(feature_source))
    spell_file = os.path.join(data_dir, 'misc/english_spellings.csv')
    spell_df = pd.read_csv(spell_file, index_col='UK')
    spell_dict = spell_df['US'].to_dict()

    # Defines functions that are from the NLTK, I assume?
    stemmer = EnglishStemmer()
    test_stemmer = PorterStemmer()

    # Cycles through each raw .txt file
    for file_ in glob(os.path.join(raw_dir, '*.txt')):
        filename = os.path.basename(file_)
        print('Processing {0}'.format(filename))
        with open(file_, 'rb') as fo:
            text = fo.read()

        text = text.decode('utf8', 'ignore').encode('ascii', 'ignore')

        # Clean text
        text = abbr.clean_str(text)

        # Detect and expand abbreviations
        text = abbr.expandall(text)

        # Remove periods (for abbreviations)
        text = text.replace('.', '')

        # Replace British words with American words.
        pattern = re.compile(r'\b(' + '|'.join(spell_dict.keys()) + r')\b')
        text = pattern.sub(lambda x: spell_dict[x.group()], text)

        # Defines stem_list which will be a list of all the words in the file,
        # not including spaces
        stem_list = []
        for word in text.split():
            # Use Porter stemmer to test for string unicode encoding, then use
            # English stemmer to perform stemming
            try:
                ' '.join(['kdkd', test_stemmer.stem(word), 'kdkd'])
            except:
                word = word.decode('utf8', 'ignore').encode('ascii', 'ignore')
            stem_list.append(stemmer.stem(word))

        # Writes the stem_list
        with open(os.path.join(stem_dir, filename), 'wb') as fo:
            fo.write(' '.join(stem_list))

        # Writes the processed text
        with open(os.path.join(proc_dir, filename), 'wb') as fo:
            fo.write(text)


def label_data(data_dir):
    """
    Creates a binary data metadata matrix
    Convert metadata files to instance-by-label matrix.
    """
    # Reads in the metadata matrices (i.e., paper by metadata matrix)
    metadata_dir = os.path.join(data_dir, 'metadata/')
    filenames = sorted(glob(os.path.join(metadata_dir, 'Karina*.csv')))

    converter = {'Paradigm Class': 'ParadigmClass',
                 'Behavioral Domain': 'BehavioralDomain',
                 'Diagnosis': 'Diagnosis',
                 'Stimulus Modality': 'StimModality',
                 'Stimulus Type': 'StimType',
                 'Response Modality': 'RespModality',
                 'Response Type': 'RespType',
                 'Instructions': 'Instruction'}

    columns = converter.keys()

    # Cycles through all columns and all metadata files and concatenates everything
    metadata_dfs = [pd.read_csv(file_, dtype=str)[['PubMed ID'] + converter.keys()] for file_ in filenames]
    metadata_df = pd.concat(metadata_dfs, ignore_index=True)

    # Transforms to table format and sorts the metadata matrices
    all_labels = []
    for column in converter.keys():
        dimension = converter[column]
        dim_labels = get_label_parents(metadata_df, column, dimension)
        all_labels += dim_labels

    all_labels = [label_con.get(label, label) for label in all_labels]
    all_labels = [label for label in all_labels if label != 'DROP']
    all_labels = sorted(list(set(all_labels)))

    # Get list of annotated papers with associated files
    metadata_df = metadata_df[metadata_df['PubMed ID'].str.contains('^\d+$')].reset_index()
    list_of_pmids = metadata_df['PubMed ID'].unique().tolist()

    column_names = ['pmid'] + all_labels
    label_df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
    label_df['pmid'] = list_of_pmids

    # This generates a binary matrix of pmids by all labels.
    for row in metadata_df.index:
        pmid = metadata_df['PubMed ID'].iloc[row]

        # Only label papers with associated files.
        if pmid in list_of_pmids:
            for column in columns:
                exp_labels = metadata_df[column].iloc[row]
                if pd.notnull(exp_labels):
                    exp_labels = exp_labels.split('| ')
                    exp_labels = ['{0}.{1}'.format(converter[column], clean_str(label)) for label in exp_labels]
                    for label in exp_labels:
                        corr_label = label_con.get(label, label)
                        for out_column in label_df.columns:
                            # Count each label toward itself and its parents.
                            if corr_label.startswith(out_column):
                                out_row = label_df.loc[label_df['pmid']==pmid].index[0]
                                label_df[out_column].iloc[out_row] = 1

    # Save DataFrames.
    count_df = label_df.sum().to_frame()
    label_df = label_df.astype(int).astype(str)
    out_file = os.path.join(data_dir, 'labels/fiu_labels.csv')
    label_df.to_csv(out_file, index=False)

    count_file = os.path.join(data_dir, 'labels/label_counts.csv')
    count_df = count_df.ix[1:]
    count_df.index.name = 'Label'
    count_df.columns = ['Count']
    count_df.to_csv(count_file, index=True)


def generate_gazetteer(data_dir):
    """
    Creates gazetteer for CogAt.
    """
    gaz_dir = os.path.join(data_dir, 'gazetteers/')
    sources = ['abstract', 'full']

    # Cognitive Atlas gazetteer
    weights = {'isSelf': 1,
               'isKindOf': 1,
               'inCategory': 1}
    vocab_df = pd.read_csv(join(ec.utils.get_resource_path(), 'ontology',
                                'unstemmed_cogat_vocabulary.csv'))
    weight_df = pd.read_csv(join(ec.utils.get_resource_path(), 'ontology',
                                 'unstemmed_cogat_weights.csv'), index_col='id')
    for source in sources:
        text_folder = join(data_dir, 'text/cleaned_{0}/'.format(source))
        out_dir = join(data_dir, 'text/cogat_cleaned_{0}/'.format(source))
        count_df = ec.extract.extract_folder(text_folder, vocab_df, stem=False,
                                             subs_folder=out_dir, abbrev=True)
        count_df.to_csv(join(data_dir, 'features/cogat_counts_{0}.csv'.format(source)))

        weighted_df = ec.extract.expand(count_df, weight_df)
        weighted_df.to_csv(join(data_dir, 'features/cogat_{0}.csv'.format(source)))

    print('Completed cogat extraction.')
