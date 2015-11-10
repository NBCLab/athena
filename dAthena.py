
import glob
import string
import os
import sys
import numpy as np
import scipy as sc
import pandas as pd
from collections import defaultdict
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

#This class contains all the abstracts or test
class Athena:
	#Called when the class is initialized
	def __init__(self):
		'''-----	Text Variables   -----'''
		dataFolder = os.path.join(os.path.expanduser("~"), "dAthena/data/")
		#Location the abstracts' plain-text are located

		self.corpus_directory = os.path.join(dataFolder, 'abstracts/*.txt')
        #self.corpus_directory = os.path.join(dataFolder, 'methods/*.txt')
		#self.corpus_directory = os.path.join(dataFolder, 'combined/*.txt')
		#self.corpus_directory = os.path.join(dataFolder, '2013_abstracts/*.txt')

		#Location of stopword list
		self.stopword_file = os.path.join(dataFolder,'misc_data/onix_stopwords.txt')
		#Total # of abstracts/methods loaded
		self.text_files_loaded = 0
		#Names of all loaded files
		self.filenames = None
		#Holds all of the plain text
		self.text_corpus = None

		'''-----	Meta-data Variables   -----'''
		#Location of meta_data
		self.meta_data_directory = os.path.join(dataFolder, 'meta_data/*.csv')
		self.meta_data = None
		#Column names we want to keep
		self.index_name = ['Year', 'First Author', 'Journal', 'PubMed ID']
		self.column_name = ['Diagnosis','Stimulus Modality','Response Modality','Response Type','Stimulus Type', 'Instructions', 'Behavioral Domain', 'Paradigm Class']
		self.column_name2 = ['all','Diagnosis','Stimulus Modality','Response Modality','Response Type','Stimulus Type', 'Instructions', 'Behavioral Domain', 'Paradigm Class']
		self.kept_columns = ['PubMed ID','Diagnosis','Stimulus Modality','Response Modality','Response Type','Stimulus Type', 'Instructions', 'Behavioral Domain', 'Paradigm Class']

		'''------ Combined Data ------ '''
		self.combined_meta_data = None
		self.combined_df = None

		'''------- Training & Test Data ---- '''
		self.train_data = None
		self.test_data = None
		self.train_label = None
		self.test_label = None

		'''------- MultiLabelBinarizer ----'''
		self.mlb = None
		self.label_mlb = None
		self.label_text_dict = None
		self.label_bin_dict = None
		self.label_bin_df = None
		self.label_df = None
		self.label_dimension_dict = None

		'''-----	Pipeline   -----'''
		self.clf_pipeline = None
		self.clfs = None
		self.clf_names = None
		
		self.results = None
		self.text = None

		self.grid_searches = None

		#Label dim stuff
		self.n_feature = None
		self.dimension_end = None
		self.dimension_beg = None
		self.test = None
		self.test2 =None

		''' Test '''
		self.bnb_clf = None
		self.tvect = None
		self.x1 = None
		self.test_pred = None
		self.test_f1 = None

	#Read in stopwords from file
	def read_stopwords(self):
		print ('Reading stopwords...')
		with open(self.stopword_file) as f:
			self.stopwords = f.read().split()
		return self

	#read_text(self) : Load in text
	# Corpus can be accessed with corpus.text_corpus['dictkey']
	# dictkeys are the filenames minus the extension
	# e.g. corpus.text_corpus['11467915'] will give you text from 11468915_a_p.txt
	def read_text(self):
		print('Reading text data...')
		temp_corpus = dict()
		for filename in sorted(glob.glob(self.corpus_directory)):
			#Read in text from file
			f = open(filename)
			text = f.read()
			f.close()
			#The [:-8] gets rid of the last 8 chars of file name
			#11467915_a_p.txt -> 11467915
			#Abstracts ext_len = -8, methods ext_len = -4
			ext_len = -8
			#temp_corpus[os.path.basename(filename)[:-8]] = text
			temp_corpus[os.path.basename(filename)[:ext_len]] = text


		#filenames now contains all the dictionary kegs (filenames)
		self.filenames = sorted(temp_corpus);
		#text_corpus contains all the plaintext keyed with filenames
		self.text_corpus = temp_corpus
		#update number of files
		self.text_files_loaded = len(self.filenames)
		print('Files loaded: '+str(self.text_files_loaded))
		return self

	#Loads in all meta-data from .csv files (pain.csv, face.csv, etc)
	def read_meta_data(self):
		print('Reading meta-data...')
		#Read in all of the meta data files
		df = [pd.read_csv(i, dtype = np.str) for i in sorted(glob.glob(self.meta_data_directory))]
		#Now we have to join all the seperate tables stored in df
		df = pd.concat(df, ignore_index = True)
		#Keep useful labels we want
		df = df.loc[:,self.index_name + self.column_name]
		#Drop duplicates
		df = df.drop_duplicates()
		#Drop rows with null PMIDs
		df = df[(df['PubMed ID']!='null')]

		#Drop rows who are missing columns
		df = df.dropna()
		df['PubMed ID'] = df['PubMed ID'].apply(int)
		#Sort the table
		df = df.sort_values(by=['PubMed ID', 'Year', 'First Author', 'Journal'])
		df['PubMed ID'] = df['PubMed ID'].apply(str)
		#print df['PubMed ID']

		self.meta_data = df
		return df

	#Merge experiments with multiple labels in one label dimension
	def _merge_series(self, series):
		'''
		Merge experiments with multiple labels in one label dimension
		'''
		label_zip =  zip(series)
		end_set = set()

		for each_val in label_zip:
			temp_val = [v.split('|') for v in each_val]
			for v in temp_val:
				v = [s.strip() for s in v]
				end_set.update(v)
		if 'None' in end_set:
			end_set.remove('None')

		return end_set

	#A lot of the meta data has multiple rows for one PMID so let's merge them
	def combine_meta_data(self):
		print('Combining meta-data...')
		#New dataframe with the index being PMIDs
		df = pd.DataFrame(index=self.meta_data['PubMed ID'].unique(), columns=self.column_name, dtype='string')

		#Loop over rows
		for row_index, current_row in self.meta_data.iterrows():
			#Grab current PMID
			current_pmid = current_row['PubMed ID']
			#Grab all rows which match this PMID (multiple rows are from same paper)
			current_record = self.meta_data[self.meta_data['PubMed ID']==current_pmid]
			#Loop over all columns we want to keep
			for curr_column in self.column_name:
				#Save unique values for each column in each PMID into df
				df.loc[current_pmid,curr_column] = self._merge_series(current_record[curr_column].unique())
		self.combined_meta_data = df
		return self

	#Combines the meta-data + abstracts into one table
	def combine_data(self):
		print('Combining abstracts + meta_data...')
		#Meta data table basis for our new combined data array
		self.combined_df = self.combined_meta_data
		#Add an abstract text column
		self.combined_df['Abstract Text']=''

		for row_index,current_row in self.combined_meta_data.iterrows():
			try:
				current_pmid = row_index
				current_abs = self.text_corpus[current_pmid]
				self.combined_df.loc[current_pmid,'Abstract Text'] = current_abs
			except Exception as e:
				#Throws key error if we didn't find the abstract text, so drop the column from the table
				#print e
				self.combined_df = self.combined_df.drop(row_index)
				pass
		#print self.combined_df
		return self


	#Grab all unique labels for a paradigm class!
	def get_unique_labels(self,paradigm_label):
		paradigm_list = []
		for i in self.combined_df[paradigm_label]:
			for c_set in i:
				paradigm_list.append(c_set)
		paradigm_set = set(paradigm_list)
		return paradigm_set

	#Splits the data into two partitions!
	def split_data(self, partition_size = 0.30):
		print('Partitioning Data...')
		self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(self.combined_df,self.label_bin_df,test_size=partition_size)
		return self

	def split_data_abs(self):
		print('Partitioning Data...')
		self.train_data = self.combined_df
		self.test_data = self.combined_df
		self.train_label = self.label_bin_df
		self.test_label = self.label_bin_df
		return self

	#Here we set up the MLB as well as dictionarys corresponding to the binary matrix
	def binarize(self):
		self.label_text_dict = defaultdict(list)

		for index, current_row in self.combined_df.iterrows():
			for current_column_name in self.column_name:
				self.label_text_dict[current_column_name].append(current_row[current_column_name])

		''' This is stupid, i am stupid
		#Generates a dictionary list for all possible labels in each column
		self.label_text_dict = defaultdict(list)
		for lbl in self.column_name:
			for set_val in self.get_unique_labels(lbl):
				self.label_text_dict[lbl].append(set_val)
		'''

		self.label_bin_dict = {key:MultiLabelBinarizer().fit_transform(label_list) for key, label_list in self.label_text_dict.items()}
		self.label_bin_df =  pd.DataFrame(np.concatenate([self.label_bin_dict[k] for k in self.column_name],1))
		self.combined_df.index = range(len(self.combined_df.index))
		self.label_df =  pd.concat([self.combined_df, self.label_bin_df], axis = 1)

		label_mlb_dict = {key:MultiLabelBinarizer().fit(label_list) for key, label_list in self.label_text_dict.items()}
		label_mlb = MultiLabelBinarizer()
		label_mlb.classes_ = np.concatenate([label_mlb_dict[key].classes_ for key in self.column_name])
		#self.label_mlb = label_mlb
		
		label_dimension_dict = {key:set(label_mlb_dict[key].classes_) for key in self.column_name}
		self.label_dimension_dict = label_dimension_dict
		
		#Conver to array!
		self.label_bin_df= self.label_bin_df.values

		#d = {'label_bin_dict': self.combined_df[0]}
		#df2 = pd.DataFrame(data = d)
		#df2.to_csv("results/test7.csv",',')
		#self.label_bin_df.to_csv("results/test7.csv",',')
		#np.savetxt('results/test.csv',self.label_bin_df,fmt='%d',delimiter=',')

		return self
	def create_pipeline_abs(self):
		self.clf_names = ['MNB','BNB','LRL1','LRL2','SVCL1','SVCL2']
		clfs = [
			MultinomialNB(alpha=0.01),
			BernoulliNB(alpha=0.01),
			LogisticRegression(C=100,penalty = 'l1', class_weight='auto'),
			LogisticRegression(C=10,penalty = 'l2', class_weight='auto'), 
			LinearSVC(C=10,penalty = 'l1', class_weight='auto', dual=False),
			LinearSVC(C=1,penalty = 'l2', class_weight='auto', dual=False)]

		ovr_clfs = [OneVsRestClassifier(clf) for clf in clfs]
		self.pipeline = [Pipeline([
						('vect',TfidfVectorizer(min_df = 3, stop_words = self.stopwords, sublinear_tf = True)),
						('ovr',clf)]) for clf in ovr_clfs]


		#So we can pick label dims!
		self.n_feature = [len(self.label_dimension_dict[col]) for col in self.column_name]
		self.dimension_end = dict(zip(self.column_name, np.cumsum(self.n_feature)))
		self.dimension_beg = {col:self.dimension_end[col]-len(self.label_dimension_dict[col]) for col in self.column_name}
		return self

	def run_grid_search_abs(self):
		label_pred = dict()
		f1_score = dict()
		for i in range(len(self.pipeline)):
			self.test = self.pipeline[i].fit(self.train_data['Abstract Text'], self.train_label)
			self.test2 = self.pipeline[i].predict(self.test_data['Abstract Text'])
			if not os.path.exists('results/'):
				os.mkdir('results')
			np.save('results/conf_'+str(i)+'.npy',self.test2)
		return self

	'''++++++++++++++++++++++++++++++++++++++++++++++'''
	def create_2013_pipeline(self, alpha_param):
		self.clf_names = ['BNB']
		clfs = [ BernoulliNB(alpha=alpha_param) ]
		#LinearSVC(penalty = 'l2', class_weight='auto', dual=False, C=1.0)]

		ovr_clfs = [OneVsRestClassifier(clf) for clf in clfs]

		#'vect',TfidfVectorizer(min_df = 3, stop_words = self.stopwords, sublinear_tf = True))
		self.pipeline = [Pipeline([
						('vect',CountVectorizer(binary=True)),
						('ovr',clf)]) for clf in ovr_clfs]

		self.n_feature = [len(self.label_dimension_dict[col]) for col in self.column_name]
		self.dimension_end = dict(zip(self.column_name, np.cumsum(self.n_feature)))
		self.dimension_beg = {col:self.dimension_end[col]-len(self.label_dimension_dict[col]) for col in self.column_name}

		self.pipeline[0].fit(self.train_data['Abstract Text'], self.train_label)
		self.test_pred = self.pipeline[0].predict(self.test_data['Abstract Text'])
		#self.test_f1 = metrics.f1_score(self.test_label,self.test_pred, average='micro')
		return self.test_f1


	#All processing will be done with pipelines to make things easier
	def create_pipeline(self):
		print('Creating pipeline...')

		#Classifier names, arbitrary
		self.clf_names = ['MNB','BNB','LRL1','LRL2','SVCL1','SVCL2']
		#Classifiers used in the experiment
		clfs = [
			MultinomialNB(),
			BernoulliNB(),
			LogisticRegression(penalty = 'l1', class_weight='auto'),
			LogisticRegression(penalty = 'l2', class_weight='auto'), 
			LinearSVC(penalty = 'l1', class_weight='auto', dual=False),
			LinearSVC(penalty = 'l2', class_weight='auto', dual=False)]

		#Generates one vs rest classifiers for each classifier
		ovr_clfs = [OneVsRestClassifier(clf) for clf in clfs]

		#Create pipeline consisting of the vectorizer and one vs rest classifiers
		self.pipeline = [Pipeline([
						('vect',TfidfVectorizer(min_df = 3, stop_words = self.stopwords, sublinear_tf = True)),
						('ovr',clf)]) for clf in ovr_clfs]
		
		#Parameters to grid search over. Look at the individual classifiers for details
		pipelines_parameters = [
			{'ovr__estimator__alpha':[0.01, 0.1, 1, 10]},
			{'ovr__estimator__alpha':[0.01, 0.1, 1, 10]}, 
			{'ovr__estimator__C':[0.1, 1, 10, 100]},
			{'ovr__estimator__C':[0.01, 0.1, 1, 10]},
			{'ovr__estimator__C':[0.01, 0.1, 1, 10]},
			{'ovr__estimator__C':[0.01, 0.1, 1, 10]}]

		#Pass above list of params to the pipeline
		self.pipelines_parameters = dict(zip(self.clf_names, pipelines_parameters))

		#define grid searches and 10-Fold validation for the pipeline
		self.grid_searches = [
			{'grid_search': GridSearchCV(pl, param_grid = pp, 
			cv = KFold(len(self.train_data['Abstract Text']), n_folds = 10, shuffle=True), scoring = 'f1_micro', n_jobs = -1, verbose = 1)} 
			for pl, pp in zip(self.pipeline, pipelines_parameters)
			]

		#Variable to hold our clf names and grid search stuff
		self.estimators = dict(zip(self.clf_names, self.grid_searches))

		#So we can pick label dims!
		self.n_feature = [len(self.label_dimension_dict[col]) for col in self.column_name]
		self.dimension_end = dict(zip(self.column_name, np.cumsum(self.n_feature)))
		self.dimension_beg = {col:self.dimension_end[col]-len(self.label_dimension_dict[col]) for col in self.column_name}

		
	''' TODO: Remove this
	def train_clfs(self):
		for clf in self.pipeline:
			clf.fit(self.train_data['Abstract Text'],self.train_label)
		return self

	def predict_clfs(self):
		self.results = self.pipeline[1].predict(self.test_data['Abstract Text'])
		return self
	'''

	def compute_2013_f1(self,label_dimension='all'):
		if label_dimension == 'all':
			return metrics.f1_score(self.test_label,self.test_pred, average='micro')
		else:
			label_index_end = self.dimension_end[label_dimension]
			label_index_beg = self.dimension_beg[label_dimension]
			return metrics.f1_score(self.test_label[:, label_index_beg:label_index_end], 
                self.test_pred[:, label_index_beg:label_index_end],average='micro')

	def get_2013_f1s(self,run_num):
		ary = np.empty([1,len(self.column_name2)])
		y=0
		for lbl in self.column_name2:
			val = self.compute_2013_f1(lbl)
			ary[0,y]= val
			y = y+1
		np.save('results/'+'f1_run'+str(run_num),ary)


	#TODO: Refactor this into one function...
	def compute_f1(self, clf_name, label_dimension = 'all'):
		label_pred = self.estimators[clf_name]['label_pred']
		return self._compute_f1(label_pred, label_dimension)

	def _compute_f1(self, label_pred, label_dimension = 'all'):
		if label_dimension == 'all':
			return metrics.f1_score(self.test_label, label_pred, average='micro')
		else:
			#Here we grab only the specific label dimension
			#This can be done because we generated those index-s
			label_index_end = self.dimension_end[label_dimension]
			label_index_beg = self.dimension_beg[label_dimension]
			return metrics.f1_score(self.test_label[:, label_index_beg:label_index_end], 
                label_pred[:, label_index_beg:label_index_end],average='micro')

	#Grid search over params to get best ones.
	def run_grid_search(self):
		label_pred = dict()
		f1_score = dict()
		for clf_name, clf in self.estimators.items():
			#print ("*** Grid Search " + clf_name + " ****")
			clf['grid_search'].fit(self.train_data['Abstract Text'], self.train_label)
			clf['label_pred'] = clf['grid_search'].predict(self.test_data['Abstract Text'])
		return self

	#Generates F1 scores and saves them
	def get_f1s(self,run_num):
		ary = np.empty([6,len(self.column_name2)])
		x = 0
		for clf in self.clf_names:
			y = 0
			for lbl in self.column_name2:
				val = self.compute_f1(clf,lbl)
				ary[x,y] = val
				y = y + 1
			x = x + 1
		np.save('results/'+'f1_run'+str(run_num),ary)
		return self

	# counts the number of words (that aren't stop words) in a body of text
	def nonstop_word_count(self, text, dic):
		words = text.split()
		for w in words:
			if w not in self.stopwords:
				if w in dic:
					dic[w] = dic.get(w, 0) + 1
				else:
					dic[w] = 1
		return dic

	# counts the number of words (that aren't stop words) across all articles
	def write_nonstop_word_count_per_article(self):
		for f in self.filenames:
			dic = defaultdict(int)
			dic = self.nonstop_word_count(self.text_corpus[f], dic)
			df = pd.DataFrame(data = dic.items())
			if not os.path.exists("./wordCount/"):
				os.mkdir("./wordCount/")
			df.to_csv("./wordCount/"+f+".csv", ',')
		
	# counts the number of words (that aren't stop words) across all articles
	def total_nonstop_word_count(self):
		dic = defaultdict(int)
		for f in self.filenames:
			dic = self.nonstop_word_count(self.text_corpus[f], dic)
		return dic
	
	#Returns a list with the number of words in each abstract
	def word_list(self):
		word_list = []
		for abs in self.label_df['Abstract Text']:
			words = abs.split()
			word_list.append(len(words))
		return word_list

	#Returns a list with the number unique words in each abstract
	def unique_word_list(self):
		word_list = []
		for abs in self.label_df['Abstract Text']:
			words = abs.split()
			words_set = set(words)
			word_list.append(len(words_set))
		return word_list

	def do_confs(self,c_run):
		for clf in self.clf_names:
			for lbl_dim in self.column_name:
				self.conf(clf,lbl_dim,c_run)
		return self

	def do_confs_abs(self,c_run):
		for clf in self.clf_names:
			for lbl_dim in self.column_name:
				self.abs_conf(clf,lbl_dim,c_run)
		return self

	def abs_conf(self,clf_name,label_dimension,c_run):
		label_pred = self.test2
		label_index_end = self.dimension_end[label_dimension]
		label_index_beg = self.dimension_beg[label_dimension]
		subset_true = self.test_label[:, label_index_beg:label_index_end]
		# make sure shape is the same
		if "Paradigm" in label_dimension:
			print(subset_true.shape)
		subset_pred = label_pred[:, label_index_beg:label_index_end]
		conf_array = np.empty(shape=subset_true.shape)
		for (x,y), value in np.ndenumerate(subset_true):
			# true negative
			if subset_true[x,y] == 0 and subset_pred[x,y] == 0:
				conf_array[x,y] = 1
			# false positive
			elif subset_true[x,y] == 0 and subset_pred[x,y] == 1:
				conf_array[x,y] = 2
			# false negative
			elif subset_true[x,y] == 1 and subset_pred[x,y] == 0:
				conf_array[x,y] = 3
			# true positive
			elif subset_true[x,y] == 1 and subset_pred[x,y] == 1:
				conf_array[x,y] = 4

		lbls = sorted(list(self.label_dimension_dict[label_dimension]))
		if not os.path.exists('results/heatmaps/'):
			os.mkdir('results/heatmaps/')
		#np.save('results/heatmaps/'+clf_name+'_'+label_dimension+'_'+str(c_run)+'.csv',conf_array)
		print('Writing results/heatmaps/'+clf_name+'_'+label_dimension+'_'+str(c_run)+'.csv')
		np.savetxt('results/heatmaps/'+clf_name+'_'+label_dimension+'_'+str(c_run)+'.csv',conf_array, fmt='%d', delimiter=',')
		
		f = open('results/heatmaps/'+clf_name+'_'+label_dimension+'_label_'+str(c_run)+'.txt', 'w')
		for item in lbls:
			f.write(item + '\n')
		f.close() 

		return self

	def conf(self,clf_name,label_dimension,c_run):
		label_pred = self.estimators[clf_name]['label_pred']
		label_index_end = self.dimension_end[label_dimension]
		label_index_beg = self.dimension_beg[label_dimension]
		subset_true = self.test_label[:, label_index_beg:label_index_end]
		subset_pred = label_pred[:, label_index_beg:label_index_end]
		conf_array = np.empty(shape=subset_true.shape)
		for (x,y), value in np.ndenumerate(subset_true):
			# true negative
			if subset_true[x,y] == 0 and subset_pred[x,y] == 0:
				conf_array[x,y] = 1
			# false positive
			elif subset_true[x,y] == 0 and subset_pred[x,y] == 1:
				conf_array[x,y] = 2
			# false negative
			elif subset_true[x,y] == 1 and subset_pred[x,y] == 0:
				conf_array[x,y] = 3
			# true positive
			elif subset_true[x,y] == 1 and subset_pred[x,y] == 1:
				conf_array[x,y] = 4
		lbls = sorted(list(self.label_dimension_dict[label_dimension]))

		if not os.path.exists('results/heatmaps/'):
			os.mkdir('results/heatmaps/')
		#np.save('results/heatmaps/'+clf_name+'_'+label_dimension+'_'+str(c_run)+'.csv',conf_array)
		np.savetxt('results/heatmaps/'+clf_name+'_'+label_dimension+'_'+str(c_run)+'.csv',conf_array, fmt='%d', delimiter=',')
		f = open('results/heatmaps/'+clf_name+'_'+label_dimension+'_label_'+str(c_run)+'.txt', 'w')
		for item in lbls:
			f.write(item + '\n')
		f.close() 

	def get_params(self,run_num):
		p_alpha = 'ovr__estimator__alpha'
		p_c = 'ovr__estimator__C'
		winning_params = []
		winning_vals = []

		#These clfs have alpha param
		for clf in ['MNB','BNB']:
			best_param = self.estimators[clf]['grid_search'].best_params_
			param_val = best_param.get(p_alpha)
			winning_params.append(p_alpha)
			winning_vals.append(param_val)

		#These clfs has c param
		for clf in ['LRL1','LRL2','SVCL1','SVCL2']:
			best_param = self.estimators[clf]['grid_search'].best_params_
			param_val = best_param.get(p_c)
			winning_params.append(p_c)
			winning_vals.append(param_val)

		f = open('results/best_params_'+str(run_num)+'.txt', 'w')
		for item in range(0,6):
			f.write(str(winning_vals[item]))
			f.write(" ")
		f.close()

		return self

	#Coeff vectors
	def get_coeff_vectors(self):
		for clf_i in range(0,6):
			#n_labels = 
			n_coef = self.pipeline[clf_i].steps[1][1].coef_.shape
			coef_vect = self.pipeline[clf_i].steps[1][1].coef_
			#intercept = self.pipeline[clf_i].steps[1][1].intercept_
			feature_list = self.pipeline[clf_i].steps[0][1].get_feature_names()
			np.savetxt("results/coef_vect_"+str(clf_i)+'.csv',coef_vect, delimiter = ',')
			f = open('results/coef_names_'+str(clf_i)+'.txt', 'w')
			for item in feature_list:
				f.write(item)
				f.write('\n')
			f.close()

			#np.savetxt("results/coef_list_"+str(clf_i)+'.csv',feature_list,delimiter=',')
		return self

	def get_246(self,lbl):
		beg_index = athena.label_dimesion_beg[lbl]
		end_index = athena.label_dimesion_end[lbl]

#Replicating matt's results from 2013 (now with 100% more countvectorization)
def run_2013_abstracts(alpha_param):
	#2013 Test run
	for run in range(0,10):
		#All same preprocessing/data stuff as normal run
		athena = Athena()
		athena.read_text()
		athena.read_stopwords()
		athena.read_meta_data()
		athena.combine_meta_data()
		athena.combine_data()
		athena.binarize()
		athena.split_data()
		#Now we use a seperate pipeline with BNB, word count vect instead of tfidf
		athena.create_2013_pipeline(alpha_param)
		#Also seperate way to get f1s obviously
		athena.get_2013_f1s(run)


# Program main functions
if __name__ == "__main__":

	#run_2013_abstracts(0.1)


	run = 0
	athena = Athena()
	athena.read_text()
	athena.read_stopwords()	
	athena.read_meta_data()
	athena.combine_meta_data()	
	athena.combine_data()

	athena.binarize()
	athena.split_data_abs()
	athena.create_pipeline_abs()
	athena.run_grid_search_abs()
	athena.do_confs_abs(run)
	athena.get_coeff_vectors()
	#athena.get_f1s(run)

	'''
	#Normal Run
	for run in range(0,10):
		print('Run '+str(run))
		athena = Athena()
		athena.read_text()
		athena.read_stopwords()
		athena.read_meta_data()
		athena.combine_meta_data()
		athena.combine_data()
		athena.binarize()
		athena.split_data()
		athena.create_pipeline()
		athena.run_grid_search()
		athena.get_f1s(run)
		athena.do_confs(run)
		athena.get_params(run)
	'''
