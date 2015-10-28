import glob
import string
from os import path

def preprocess_file(file_path,save_path,additional_ext=''):
	file_name = path.basename(filepath)
	dir_name = path.dirname(filepath)

	#Read in file (lower case)
	f = open(file_path)
	text = f.read().lower()
	f.close()

