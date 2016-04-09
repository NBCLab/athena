import glob
from os import path
import shutil
import numpy as np

abs_dir = 'data/abstracts/*.txt'
methods_dir = 'data/methods/*.txt'

'''
abs_lengths = []
for filename in glob.glob(methods_dir):
	f = open(filename)
	text = f.read()
	n_words = len(text.split())
	abs_lengths.append(n_words)
	print n_words

'''

abs_list = []
met_list = []

for filename in glob.glob(methods_dir):
	f = open(filename)
	text = f.read()
	f.close()

	temp_list = []

	for i in text.split():
		temp_list.append(i)

	unique = len(set(temp_list))

	abs_list.append(unique)

print (abs_list)





'''
for filename in glob.glob(methods_dir):
	f = open(filename)
	text = f.read()
	f.close()
	for i in text.split():
		met_list.append(i)

print 'abs'+str(len(set(abs_list)))
print 'met'+str(len(set(met_list)))
'''