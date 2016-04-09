import numpy as np
import scipy.sparse
import pickle
import pandas as pd
from scipy import stats

'''Use either 'abstracts','methods',or 'combined'''
label_dir = 'methods/'

#Dictionary to match file number to classifier.
clfs = {'BNB':0,'MNB':1,'LRL1':2,'LRL2':3,'SVCL1':4,'SVCL2':5}

#List of our label names
label_names = ['Diagnosis','Stimulus Modality','Response Modality','Response Type','Stimulus Type', 'Instructions', 'Behavioral Domain', 'Paradigm Class']

#This holds the label dimension dictionary and tells us how to subset the dictionary
#to get the proper names for a specific label_name
label_dimension_dict = pickle.load(open('results/'+label_dir+'label_dimension_dict.p','rb'))
label_dimension_beg = pickle.load(open('results/'+label_dir+'dim_beg.p','rb'))
label_dimension_end = pickle.load(open('results/'+label_dir+'dim_end.p','rb'))
#Get word counts from count vectorizer for some double checking 
word_counts = pickle.load(open('word_counts/'+label_dir[:-1]+'_counts.p','rb'))



def get_lbl_range(label):
	beg = label_dimension_beg[label]
	end = label_dimension_end[label]
	return label,beg,end


# Purpose    - Loads all coefficient names for coef_ matrix, should be same for all clfs
# Parameters - clf (string) - a string from clfs list to select classifier
# Return     - list of all coefficient names
def get_coef_names(clf):
	clf_num = clfs[clf]
	vocab = pickle.load(open('results/'+label_dir+'vocab_'+str(clf_num)+'.p','rb'))
	return sorted(vocab.keys())


# Purpose    - Loads a dict with coef names and counts
# Parameters - clf (string) - a string from clfs list to select classifier
# Return     - dict of coef names : coef counts
def get_coef_counts(clf):
	clf_num = clfs[clf]
	return pickle.load(open('results'+label_dir+'/vocab_'+str(clf_num)+'.p','rb'))

# Purpose - Turn numpy array to pandas data frame
# Params  - data(array) the data matrix
#           column,index (strings) obvious
# Return  - pandas dataframe
def to_pandas(array,clmn,idx):
	df = pd.DataFrame(data=array,columns=clmn,index=idx)
	return df

#Purpose - Turn pandas dataframe into numpy matrix
def to_numpy(pdf):
	return pdf.as_matrix()


# Purpose    - Gets coef vector for a specific label dimension, also used to generate
#              the total coef matrix for all label dimensions
# Parameters - clf (string) - which classifier from clfs list e.g. 'MNB'
#              lbl (string) - which label dimension from label_names list
# Return     - pandas dataframe containing labeled coef_ vector for label
def get_label_coef(clf,lbl):
	#Clf name to number
	clf_num = clfs[clf]
	#Open total coef matrix
	total_coef_matrix = np.loadtxt('results/'+label_dir+'coef_vect_'+str(clf_num)+'.csv',delimiter=',')
	#Get the specific elements for this label dimension
	label_range = get_lbl_range(lbl)
	#Sub matrix for just this label dimension
	sub_coef_matrix = total_coef_matrix[label_range[1]:label_range[2]]
	#Grabs dictionary for y labels
	label_dict = sorted(label_dimension_dict[lbl])
	#Every unique word pretty much
	coef_names = get_coef_names(clf)
	#Convert to pandas df
	df = to_pandas(sub_coef_matrix,coef_names,label_dict)
	return df

# Purpose    - Generates total coefficient matrix with labels
# Parameters - clf (string) - which classifier from clfs list e.g. 'MNB'
# Return     - pandas dataframe containing labeled total coef_ vector 
def generate_total_coef(clf):
	#List to hold all dataframes
	df_list = []
	#Loop over all label names and append to list
	for lbl in label_names:
		df_list.append(get_label_coef(clf,lbl))
	#Merge all sub-matricies
	df = pd.concat(df_list)
	return df

#Grab a specific row, pass in matrix and row_index
#e.g. row_index = 'Faces' or row_index = 'n-back'
def select_row(total_coef_matrix,row_index):
	return total_coef_matrix.loc[row_index]

def threshold_coef_vector(coef_matrix,min_val=None,max_val=None,replace_val=0):
	thresholded_coef = stats.threshold(coef_matrix, threshmin=min_val, threshmax=max_val, newval=replace_val)
	return thresholded_coef

def binarize_threshold(coef_matrix,minimum):
	#Replace all values under the threshold with a 0
	min_clipped = threshold_coef_vector(coef_matrix,min_val=minimum,replace_val=0)
	#for all x where 0<x set x = 1
	min_clipped[min_clipped>0] = 1
	#Turn back into pandas dataframe
	df = to_pandas(min_clipped,coef_matrix.columns,coef_matrix.index)
	return df


#input a column name  and return a dimension name e.g. Faces -> Stimulus Type
def get_dimension_name(column_name):
	found_dimension = []
	#since .iteritems isnt working i'll just do it this way!
	keys = label_dimension_dict.keys()
	#Loop over every key in dict (aka every label dimension)
	for dimension in keys:
		#search every label in the dimension
		for label in label_dimension_dict[dimension]:
			if label == column_name:
				found_dimension.append(dimension)

	#This should NEVER happen but just incase..
	if len(found_dimension) > 1:
		print "WARNING: Found multiple dimension names for this label!"

	return found_dimension[0]

def append_coef_count(clf,matrix):
	index = matrix.index
	index_list = []
	for idx in index:
		index_list.append(get_coef_counts(clf)[idx])
	idx_series = pd.Series(index_list, index=index)
	matrix['Word Count'] = idx_series
	return matrix

#Returns counts of word 
def get_word_count(word):
	count = word_counts[word]
	return count

#Takes pandas series of a row and outputs a pandas dataframe which includes coef
#and word count
def append_word_count(pseries):
	name = pseries.name
	output_matrix = pseries.to_frame()
	#get all the words in this column
	index_list = list(faces_row_positive.index)

	#loop over all words in index and grab their counts
	index_count = []
	for c_word in index_list:
		index_count.append(word_counts[c_word])

	output_matrix[name+' Count'] = pd.Series(index_count, index=output_matrix.index)

	return output_matrix


#Example usage!

#Get coefficient matrix for 'Diagnosis' Label dimension (LRL1)
diagnosis_df = get_label_coef('LRL1','Diagnosis')

#Get total coefficient matrix for LRL1 classifier
total_dataframe = generate_total_coef('LRL1')
#Get absolute values of coefficients
absolute_dataframe = abs(total_dataframe)

binarize_dataframe = binarize_threshold(absolute_dataframe,0)

#Select 'Faces' row (use df.name to return faces) (label dimension dict has all possible values e.g. Faces)
faces_row = select_row(total_dataframe,'Faces')
#Select faces row where abs(coef) > 0
faces_row_positive = faces_row[abs(faces_row)>0]
#Number of non zero elements in faces
faces_non_zero = len(faces_row_positive)
#Find out which label dimension 'Faces' is in
faces_dimension = get_dimension_name(faces_row.name)

#Append word counts to faces_row_positive and get a new dataframe
faces_positive_counts = append_word_count(faces_row_positive)


#if we want faces_row_positive to be a dataframe instead of a pandas.series we use..
#faces_row_positive = faces_row_positive.to_frame()



