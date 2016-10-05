# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 00:07:14 2016

Prepare data for extraction and analysis.

I want to add fuzzy wuzzy

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
import re
import pandas as pd
import gazetteers
import cogat
import references
from glob import glob
import random
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import PorterStemmer
from utils import cogpo_columns, clean_str, df_to_list
from abbreviation_extraction import PhraseFinder
import feature_extraction
import cPickle as pickle


# Defines function process_raw_data which works on the athena data directory
def process_raw_data(data_dir="/home/data/nbc/athena/athena-data/"):
    """
    Label data using metadata files, split data into training and test
    datasets.
    """

#   Calls the process_corpus function, defined below
#   process_corpus reads in the text, performs abbreviation, spelling, translation, and overall text Processing
#   process_corpus outputs the processed text for each file and the stemmed file
#    process_corpus(data_dir)

#   Calls the label_data function, defined below
#   label_data reads in the metadata csv files, concatenates them, then reads in the processed text files
#   label_data outputs a binary pmid by label metadata matrix
#    label_data(data_dir)

#    labels_file = os.path.join(data_dir, "labels/full.csv")

#   Calls the split_data function, defined below
#   split_data reads in the full metadata matrix
#   split_data outputs a test and training dataset
#    split_data(labels_file, test_percent=0.33)

#   Calls the generate_gazetteers function, defined below
#   generate_gazetteers reads in the training metadata matrix, and some other files depending on the type of gazetteer
#   NBOW is gazetteer of choice right now
#   generate_gazetteers outputs a .csv and .pkl file containing the naive bag of words and NBOW vector
    generate_gazetteers()

#   Calls feature_extraction.py script, which is independent of this file
#   feature_extraction reads in the naive bag of words
#   feature_extraction creates separate NBOW density matrices for the training and test datasets (density wrt occurences in a single PMID file)
    feature_extraction.extract_features()
    import summarize_datasets

# Performs all the data pre-processing, reading in text, removing some abbreviations, text removal, writing out stem files and processed text
def process_corpus(data_dir="/home/data/nbc/athena/athena-data/"):
    raw_dir = os.path.join(data_dir, "text/combined/")
    stem_dir = os.path.join(data_dir, "text/stemmed_combined/")
    proc_dir = os.path.join(data_dir, "text/processed_combined/")
    spell_file = os.path.join(data_dir, "misc/english_spellings.csv")
    spell_df = pd.read_csv(spell_file, index_col="UK")
    spell_dict = spell_df["US"].to_dict()

    # Defines functions that are from the NLTK, I assume?
    stemmer = EnglishStemmer()
    test_stemmer = PorterStemmer()

    # Cycles through each raw .txt file
    for file_ in glob(os.path.join(raw_dir, "*.txt")):
        filename = os.path.basename(file_)
        print("Processing {0}".format(filename))
        with open(file_, "rb") as fo:
            text = fo.read()

        text = text.decode("utf8", "ignore").encode("ascii", "ignore")

        # Detect and expand abbreviations
        p = PhraseFinder()
        p.setup(text)

        # Combine newline-split words (denoted by a space-hyphen-newline combo)
        text = re.sub("-\s[\r\n\t]+", "", p.fullText, flags=re.MULTILINE)

        # Remove newlines and extra spaces
        text = re.sub("[\r\n\s\t]+", " ", text, flags=re.MULTILINE).lower()

        # Remove periods (for abbreviations)
        text = text.replace(".", "")

        # Replace British words with American words.
        pattern = re.compile(r"\b(" + "|".join(spell_dict.keys()) + r")\b")
        text = pattern.sub(lambda x: spell_dict[x.group()], text)

        # Defines stem_list which will be a list of all the words in the file, not including spaces
        stem_list = []
        for word in text.split():
                # Use Porter stemmer to test for string unicode encoding, then use English stemmer to perform stemming
                try:
                    test = " ".join(["kdkd", test_stemmer.stem(word), "kdkd"])
                except:
                    word = word.decode("utf8", "ignore").encode("ascii", "ignore")
                stem_list.append(stemmer.stem(word))

        # Writes the stem_list
        with open(os.path.join(stem_dir, filename), "wb") as fo:
            fo.write(" ".join(stem_list))

        # Writes the processed text
        with open(os.path.join(proc_dir, filename), "wb") as fo:
            fo.write(text)

        #Taylor, what differs between the two files just written? From a first glance, they look the same
        #Also, how much examination has there been of the files after processing? I feel like some of the truncating being performed is actually limting some intelligible words i.e., "working memory" becomes "work memory" and "work memori" (removing ze from memorize)

# Creates a binary data metadata matrix
def label_data(data_dir="/home/data/nbc/athena/athena-data/"):
    """
    Convert metadata files to instance-by-label matrix.
    """
    # Reads in the metadata matrices (i.e., paper by metadata matrix)
    metadata_dir = os.path.join(data_dir, "metadata/")
    filenames = sorted(glob(os.path.join(metadata_dir, "*.csv")))
    text_dir = os.path.join(data_dir, "text/processed_combined/")

    columns = ["Paradigm Class", "Behavioral Domain", "Diagnosis",
               "Stimulus Modality", "Stimulus Type", "Response Modality",
               "Response Type", "Instructions", "Context"]

    # Calls a function from utils.py, not a big fan of that unless this function is called in other scripts
    # Also, the function looks like it just truncates the spaces in the terms, is a separate function really necessary for that?
    column_to_cogpo = cogpo_columns(columns)

    # Cycles through all columns and all metadata files and concatenates everything
    full_cogpo = []
    metadata_dfs = [pd.read_csv(file_, dtype=str)[["PubMed ID"] + column_to_cogpo.keys()] for file_ in filenames]
    metadata_df = pd.concat(metadata_dfs, ignore_index=True)

    # Transforms to table format and sorts the metadata matrices
    for column in column_to_cogpo.keys():
        table = df_to_list(metadata_df, column, column_to_cogpo[column])
        full_cogpo += table

    full_cogpo = sorted(list(set(full_cogpo)))

    # Preallocate label DataFrame
    metadata_df = metadata_df[metadata_df["PubMed ID"].str.contains("^\d+$")].reset_index()
    list_of_pmids = metadata_df["PubMed ID"].unique().tolist()
    list_of_files = [os.path.splitext(file_)[0] for file_ in os.listdir(text_dir)]
    list_of_files = sorted(list(set(list_of_files)))
    list_of_pmids = sorted(list(set(list_of_pmids).intersection(list_of_files)))

    column_names = ["pmid"] + full_cogpo
    label_df = pd.DataFrame(columns=column_names, data=np.zeros((len(list_of_pmids), len(column_names))))
    label_df["pmid"] = list_of_pmids

    # This generates a binary matrix of pmids by all labels
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
    out_file = os.path.join(data_dir, "labels/full.csv")
    label_df.to_csv(out_file, index=False)

    count_file = os.path.join(data_dir, "labels/labels.csv")
    count_df = count_df.ix[1:]
    count_df.index.name = "Label"
    count_df.columns = ["Count"]
    count_df.to_csv(count_file, index=True)

# I dont think this gets used in the current script, if used elsewhere, define elsewhere or consider deleting
def resplit(labels_file, name_files):
    """
    Using existing train and test label files (to get the PMIDs) and existing
    labels file for all instances, create new train and test label files with
    new labels.
    """
    df = pd.read_csv(labels_file)
    for fi in name_files:
        df2 = pd.read_csv(fi)
        out_df = df[df['pmid'].isin(df2["pmid"])]
        out_df.to_csv(fi, index=False)

# Splits the data into test and training datasets
def split_data(labels_file, test_percent=0.33):
    """
    Find acceptable train/test data split. All labels must be represented in
    both datasets.
    """
    data_dir = os.path.dirname(labels_file)

    all_labels = pd.read_csv(labels_file)

    column_names = all_labels.columns.values
    all_data = all_labels.as_matrix()
    de_ided = all_data[:, 1:]

    n_instances = all_data.shape[0]
    index = range(n_instances)

    # Finds how many papers constitute 33% of the dataset
    n_test = int(n_instances * test_percent)
    n_train = n_instances - n_test
    print("Size of test dataset: {0}".format(n_test))
    print("Size of training dataset: {0}".format(n_train))

    split_found = False
    while not split_found:
        shuf_index = index[:]
        random.shuffle(shuf_index)
        test_rows = de_ided[shuf_index[:n_test]]
        train_rows = de_ided[shuf_index[n_test:]]

        # Ensures that at least 10 instances of each metadata field are present in the test and training sample
        # Any reason for the 10? Can that be increased? Is there any logic or way to create the most maximally different datasets?
        if np.all(np.sum(train_rows, axis=0)>10) and np.all(np.sum(test_rows, axis=0)>10):
            split_found = True

    # Writes out a binary test and training metadata matrix
    train_data = all_data[sorted(shuf_index[n_test:]), :]
    test_data = all_data[sorted(shuf_index[:n_test]), :]
    df_train = pd.DataFrame(columns=column_names, data=train_data)
    df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    df_test = pd.DataFrame(columns=column_names, data=test_data)
    df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)

# Will call the gazetteers.py script to generate the appropriate gazetteers
# For now, naive bag of words (nbow) is the gazetteer of choice
def generate_gazetteers(data_dir="/home/data/nbc/athena/athena-data/"):
    """
    Creates list of unique terms for four gazetteers derived from metadata
    available through PubMed:
        - Authors and year of publication
        - Journal of publication
        - Words in title (not including stopwords)
        - Author-generated keywords (if available on PubMed)
            - This includes multiword expressions.
    """
    label_file = os.path.join(data_dir, "labels/train.csv")
    gaz_dir = os.path.join(data_dir, "gazetteers/")
    text_dir = os.path.join(data_dir, "text/")
    stem_text_dir = os.path.join(text_dir, "stemmed_combined/")
    proc_text_dir = os.path.join(text_dir, "processed_combined/")
#    ref_text_dir = os.path.join(text_dir, "reference_data/")

    df = pd.read_csv(label_file)
    pmids = df["pmid"].astype(str).tolist()

#    metadata_gazs = gazetteers.generate_metadata_gazetteers(pmids)
#    authoryear_gaz, journal_gaz, keyword_gaz, titleword_gaz = metadata_gazs
#    gazetteers.save_gaz(authoryear_gaz, gaz_dir, "authoryear")
#    gazetteers.save_gaz(journal_gaz, gaz_dir, "journal")
#    gazetteers.save_gaz(titleword_gaz, gaz_dir, "titlewords")
#    gazetteers.save_gaz(keyword_gaz, gaz_dir, "keywords")
#    print("Completed metadata gaz.")

#    # NBOW gazetteer
#   Generates naive bag of words gazetteer on stemmed_combined from training dataset
    nbow_gaz, nbow_tfidf = gazetteers.generate_nbow_gazetteer(pmids, stem_text_dir)
    gazetteers.save_gaz(nbow_gaz, gaz_dir, "nbow")
    with open(os.path.join(gaz_dir, "nbow.pkl"), "wb") as fo:
        pickle.dump(nbow_tfidf, fo)
    print("Completed nbow gaz.")

    # Cognitive Atlas gazetteer
#    cogat_df, cogat_tfidf = cogat.create_id_sheet(pmids, proc_text_dir)
#    cogat_df.to_csv(os.path.join(gaz_dir, "cogat.csv"), index=False)
#    with open(os.path.join(gaz_dir, "cogat.pkl"), "wb") as fo:
#        pickle.dump(cogat_tfidf, fo)
#    rel_df = cogat.create_rel_sheet(cogat_df)
#    rel_df.to_csv(os.path.join(gaz_dir, "cogat_relationships.csv"), index=False)
#
#    weighting_schemes = ["ws2_up", "ws2_down", "ws2_side"]
#    for weighting_scheme in weighting_schemes:
#        weight_df = cogat.weight_rels(rel_df, weighting_scheme)
#        weight_df.to_csv(os.path.join(gaz_dir, "cogat_weights_{0}.csv".format(weighting_scheme)),
#                         index=True)
#    print("Completed cogat gaz.")

#    references_df = references.generate_references_gazetteer(pmids, ref_text_dir)
#    print("Completed references gaz.")

    # Save gazetteers
#    references_df.to_csv(os.path.join(gaz_dir, "references.csv"), index=False)
