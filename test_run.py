''' runs ATHENA '''

import data_preparation
import feature_selection
import feature_extraction
import convert_data_to_weka
import convert_clf_output
import evaluate_classifiers
from glob import glob
import os
data_dir = "/home/data/nbc/athena/athena-data/"

data_preparation.process_raw_data()
data_preparation.generate_gazetteers()
feature_extraction.extract_features()
feature_selection.run_feature_selection()
convert_data_to_weka.test()
# runs the grid search to choose parameters using MEKA classifiers
execfile("optimize_parameters.py")

model_list = glob(os.path.join(data_dir, "predictions/meka*.csv")
convert_clf_output.majority_vote(model_list, os.path.join(data_dir, "predictions/majority_vote.csv"))
evaluate_classifiers.test()
