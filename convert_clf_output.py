"""
Created on Mar 24, 2016

Takes output from MEKA and MULAN classifiers and converts them to predicted
label matrices.

@author: Jason
"""

import re
import numpy as np
import os
from scipy.stats.mstats import mode

# Some constants
re_label_count = re.compile("Label indices\s+\[(\s+(\d+)\s*)+\]", re.MULTILINE)
re_case_count = re.compile("N=(\d+)", re.MULTILINE)
re_prediction_section = re.compile("====>\n(.*)\|==============================<", re.MULTILINE|re.DOTALL)
re_array = re.compile("\|\s*\d+\s*\[.*\]\s*(\[.*\])")


def get_data(filename):
    """
    Return file contents in string.
    """
    with open(filename, "r") as fo:
        data_string = ""
        for line in fo:
            data_string += line
    return data_string


def get_meka_acc(filename):
    """
    Grab accuracy from MEKA output.
    """
    re_acc = re.compile("Accuracy\s*(\d+(\.\d+)?)", re.MULTILINE)
    data_string = get_data(filename)
    return float(re.search(re_acc, data_string).group(1))
   

def convert_meka(meka_file):
    """
    Convert classifier output from MEKA to sparse matrix and output to csv.
    
    Assumes that MEKA output is named according to [model_name].csv convention.
    """
    file_dir = os.path.dirname(meka_file)
    filename = os.path.splitext(os.path.basename(meka_file))[0]
    data_string = get_data(meka_file)
    
    prediction_section = re.search(re_prediction_section, data_string).group(1)
    labels = re.search(re_label_count, data_string)
    if labels:
        label_count = int(labels.group(2)) + 1
    
    cases = re.search(re_case_count, data_string)
    if cases:
        case_count = int(cases.group(1))
    
    # There may be a more eloquent way to handle this if statement.
    if label_count == 0 or case_count == 0:
        print("Exiting: # labels: {0}, # cases: {1}".format(label_count, case_count))
        return False
    
    predictions = np.zeros((case_count, label_count))
    lines = prediction_section.split("\n")
    for i, line in enumerate(lines):
        match = re.search(re_array, line)
        if match:
            string_array = match.group(1)
            arr = eval(string_array)
            predictions[i, arr] = 1
    np.savetxt(os.path.join(file_dir, filename+".csv"), predictions, "%d", ",")
    return predictions


def majority_vote(predictions_files, out_file):
    """
    Implement simple majority vote ensemble across individual models'
    predictions.
    """
    matrices = [np.loadtxt(file_, dtype=int, delimiter=",") for file_ in predictions_files]
    matrices = [np.expand_dims(mat, 2) for mat in matrices]
    
    matrix = np.concatenate(matrices, axis=2)
    predictions = np.asarray(mode(matrix, axis=2).mode.squeeze(), dtype=int)
    np.savetxt(out_file, predictions)

