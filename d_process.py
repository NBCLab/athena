import glob
import string
from os import path

#Function for preprocessing an individual file
def preprocess_file(file_path,save_path,additional_ext=''):
	#Get the files name
	file_name = path.basename(filepath)
	#Get the files directory
	dir_name = path.dirname(filepath)

	#Read in file (lower case) and close
	f = open(file_path)
	text = f.read().lower()
	f.close()

	

