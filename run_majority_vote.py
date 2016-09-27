# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:22:59 2016

Create majority vote file.

@author: tsalo006
"""

import os
from glob import glob
from convert_clf_output import majority_vote

pred_dir = "/home/data/nbc/athena/athena-data/predictions/"
stats_dir = "/home/data/nbc/athena/athena-data/statistics/"
pred_files = glob(os.path.join(pred_dir, "meka*.csv"))
maj_file = os.path.join(pred_dir, "majority_vote.csv")

majority_vote(pred_files, maj_file)
