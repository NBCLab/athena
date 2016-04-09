from __future__ import print_function
import re
import glob
import string
from os import path
from sklearn.feature_extraction.text import CountVectorizer

def word_break_cat(matchobj):
    return matchobj.group(1) + matchobj.group(2)
    
def process(filedir = 'data/2013_abstracts/*.txt'):
    for filepath in glob.glob(filedir):
        dirpath = path.dirname(filepath)
        filename = path.basename(filepath)
        f = open(filepath)
        text = f.read().lower()
        f.close()
        text = re.sub(r'[^a-zA-Z()[]_-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b(\w+)-\s+(\w+)\b', word_break_cat, text)
        text = re.sub(r'\bhttp.*? ',' ', text)
        text = re.sub(r'[^a-zA-Z-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        filerootname = filename.split('.')[0]
        text_process_file = open(path.join(dirpath, filerootname + '_p.txt'), 'w')
        text_process_file.write(text)