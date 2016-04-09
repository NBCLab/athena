import glob
from os import path
import shutil
import numpy as np
import pickle
#Thank you stack overflow!
from collections import Counter

abs_dir = 'data/abstracts/*.txt'
methods_dir = 'data/methods/*.txt'
combined_dir = 'data/combined/*.txt'

stemmed_abs_dir = 'data/stemmed/abstracts/*.txt'
stemmed_methods_dir = 'data/stemmed/methods/*.txt'
stemmed_combined_dir = 'data/stemmed/combined/*.txt'

def count_words(text_directory,save_dir):
	#Empty list to store all words
	word_list = []
	#Loop over all files in given directory
	for filename in glob.glob(text_directory):
		#Store all text into list
		f = open(filename)
		text = f.read()
		f.close()

		#Split all text files into individual words
		for word in text.split():
			word_list.append(word)

	word_freq = Counter(word_list)

	pickle.dump( word_freq, open(save_dir, "wb" ) )


count_words(abs_dir,'word_counts/abs_counts.p')
count_words(methods_dir,'word_counts/methods_counts.p')
count_words(combined_dir,'word_counts/combined_counts.p')
count_words(stemmed_abs_dir,'word_counts/stemmed_abs_counts.p')
count_words(stemmed_methods_dir,'word_counts/stemmed_methods_counts.p')
count_words(stemmed_combined_dir,'word_counts/stemmed_combined_counts.p')