import glob
import string
import os
import sys
import numpy as np
import scipy as sc
import pandas as pd
import pickle
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
from sklearn.externals import joblib

class AthenaClient:
	def __init__(self,data_dir="data/"):
		self.data_dir = data_dir
		self.pipeline = None

		#All data types used in experiment
		self.data_types = ['abstracts','methods','combined','s_abstracts','s_methods','s_combined']

		#All classifier names in the pipeline
		self.clfs = ['MNB','BNB','LRL1','LRL2','SVCL1','SVCL2']

		#Metadata dimensions
		self.metadata_dimensions = ['Diagnosis','Stimulus Modality','Response Modality','Response Type','Stimulus Type', 'Instructions', 'Behavioral Domain', 'Paradigm Class']

		#This dictionary relates classifier names to pipeline indices
		self.clf_dict = dict([('MNB',0),('BNB',1),('LRL1',2),('LRL2',3),('SVCL1',4),('SVCL2',5)])

		#Label dictionary and index for specific labels
		self.dim_beg = None
		self.dim_end = None
		self.dim_dict = None

		#Words used by classifier
		self.feature_names = None

		#Load in the pickled pipeline file
		#self._loadPipeline()

	#------------------------------------------------------------------------------------#
	#Loads pickled pipeline
	#data_type = 'abstracts','methods',etc
	def _loadPipeline(self,data_type):
		self.pipeline = pickle.load(open(self.data_dir+data_type+'/'+data_type+"_pipeline.p", "rb" ) )

	#Load vocabulary for a data_type, this contains all meta_data dimensions
	#data_type = 'abstracts','methods',etc
	def _loadVocab(self,data_type):
		self.dim_beg  = pickle.load(open(self.data_dir+'vocab/'+data_type+'/dim_beg.p','rb'))
		self.dim_end  = pickle.load(open(self.data_dir+'vocab/'+data_type+'/dim_end.p','rb'))
		self.dim_dict = pickle.load(open(self.data_dir+'vocab/'+data_type+'/label_dimension_dict.p','rb'))

	#This gets the feature names from the pipeline (i.e. the words the classifier uses)
	def _getFeatureNames(self,clf):
		self.feature_names = self.pipeline[self.clf_dict[clf]].steps[0][1].get_feature_names()
		return self.feature_names
	#------------------------------------------------------------------------------------#

	def _loadFile(self,file_dir):
		data = None
		with open(file_dir, 'r') as f:
			print file_dir
			data = f.read()
		return data

	def _predict(self,clf,data):
		prediction = self.pipeline[self.clf_dict[clf]].predict(data)
		return prediction

	def StartPipeline(self,data_type,classifier):
		self._loadPipeline(data_type)
		self._loadVocab(data_type)
		self._getFeatureNames(classifier)



client = AthenaClient()
client.StartPipeline("abstracts","LRL1")
#make sure its a list!!! []
#x=client._predict('BNB',client._loadFile('abstracts/9228523_a_p.txt'))
