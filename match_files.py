import glob
from os import path
import shutil

abs_dir = 'data/abstracts/*.txt'
abs_ext_len = 8
abs_files = []
abs_list = []

methods_dir = 'data/methods/*.txt'
methods_ext_len = 4
methods_files = []
methods_list = []

for filename in glob.glob(abs_dir):
	abs_files.append(filename)
	abs_list.append(path.basename(filename)[:-abs_ext_len])

for filename in glob.glob(methods_dir):
	methods_files.append(filename)
	methods_list.append(path.basename(filename[:-methods_ext_len]))

matched = list(set(methods_list) & set(abs_list))

for item in matched:
	shutil.copyfile('data/abstracts/'+item+'_a_p.txt','data/m_abstracts/'+item+'_a_p.txt')