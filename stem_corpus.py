# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:30:19 2016

Stem full text files for ATHENA.

@author: salo
"""

import os
import glob
from nltk.stem.porter import PorterStemmer

full_dir = "/home/data/nbc/athena/athena-data/text/full/"
stem_dir = "/home/data/nbc/athena/athena-data/text/stemmed_full/"

stemmer = PorterStemmer()

for file_ in glob.glob(os.path.join(full_dir, "*.txt")):
    filename = os.path.basename(file_)
    print("Stemming {0}".format(filename))
    with open(file_, "r") as fo:
        text = fo.read()

    stem_list = []
    for word in text.split():
        stem_list.append(stemmer.stem(word))
        
    with open(os.path.join(stem_dir, filename), "w") as fo:
        fo.write(" ".join(stem_list))
