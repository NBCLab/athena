import numpy as np
import pickle


#Make a dict to make function calls easier.
clfs = {'BNB':0,'MNB':1,'LRL1':2,'LRL2':3,'SVCL1':4,'SVCL2':5}

label_names = ['Diagnosis','Stimulus Modality','Response Modality','Response Type','Stimulus Type', 'Instructions', 'Behavioral Domain', 'Paradigm Class']
label_dimension_dict = pickle.load(open('results/label_dimension_dict.p','rb'))
label_dimension_beg = pickle.load(open('results/dim_beg.p','rb'))
label_dimension_end = pickle.load(open('results/dim_end.p','rb'))


def get_lbl_range(label):
	beg = label_dimension_beg[label]
	end = label_dimension_end[label]
	return label,beg,end

#Consider sorting this return value
def get_sub_labels(label):
	return label_dimension_dict[label]

#Call this using a classifier name.
def get_coef_names(clf):
	clf_num = clfs[clf]
	coef_names_list = []
	f = open('results/coef_names_'+str(clf_num)+'.txt', 'rb')
	for line in f:
		coef_names_list.append(line[:-2])#Get rid of newline characters
	f.close()
	return coef_names_list

#Get coeff vectors for specific label and classifier
def get_label_coef(clf,lbl):
	#Clf name to number
	clf_num = clfs[clf]
	#Open total coef matrix
	total_coef_matrix = np.loadtxt('results/coef_vect_'+str(clf_num)+'.csv',delimiter=',')
	#Get the specific elements for this label dimension
	label_range = get_lbl_range(lbl)
	#Sub matrix for just this label dimension
	sub_coef_matrix = total_coef_matrix[label_range[1]:label_range[2]]
	#Grabs dictionary for y labels
	label_dict = sorted(label_dimension_dict[lbl])
	#Every unique word pretty much
	coef_names = get_coef_names(clf)

	return sub_coef_matrix,coef_names,label_dict

#Example for get_label_coef
results = get_label_coef('LRL1','Diagnosis')
results[0] #(matrix)Contains the coef matrix
results[1] #(list)  Contains all 3075 words for the x axis
results[2] #(list)  Contains the label names for the y axis e.g. 'Alcoholism',



