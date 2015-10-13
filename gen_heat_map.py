import numpy as np
import matplotlib.pyplot as plt


def gen_map(adir,ylab,clf, paradigm):
	#Confusion matrix
	conf = np.load(adir+'heatmaps/'+clf+'_'+paradigm+'_0.csv.npy')
	#Hold our labels
	labels = []
	#Open labels file and add to lists
	with open(adir+'heatmaps/'+clf+'_'+paradigm+'_label_0.txt') as f:
		for line in f:
			labels.append(line)

	#Change size of figure
	plt.figure(figsize=(20,10))
	#Add some space for our bottom labels
	plt.subplots_adjust(bottom=0.20)
	#Set our x&y labels and title
	plt.title(paradigm)
	plt.ylabel(ylab)
	plt.xlabel(paradigm)
	#Set xticks to the labels we loaded
	#plt.xticks(np.arange(conf.shape[1]),labels,rotation = 90, horizontalalignment = 'center')
	plt.xticks(np.linspace(start=0.5,stop=conf.shape[1]-0.5,num=conf.shape[1]),labels,rotation = 90, horizontalalignment = 'center')
	#Plot the conf matrix
	plt.pcolor(conf,cmap=plt.cm.jet)
	#Match plot size to array
	plt.xlim([0,conf.shape[1]])
	plt.ylim([0,conf.shape[0]])
	#Add colorbar
	plt.colorbar()
    #Save image
	plt.savefig('heatmaps/'+adir+clf+'_'+paradigm+'.png')
	plt.close()

def gen_all_maps():
	clfs = ['BNB','LRL1','LRL2','SVCL1','SVCL2']
	labels = ['Diagnosis','Stimulus Modality','Response Modality','Response Type','Stimulus Type', 'Instructions', 'Behavioral Domain', 'Paradigm Class']
	types = ['abstracts','methods']

	#Do for methods and abstracts
	for paper_type in types:
		#Loop over classifiers
		for clf in clfs:
			#Loop over all labels
			for label in labels:
				#Generate the heatmap
				gen_map(paper_type+'/',paper_type,clf,label)

gen_all_maps()


#Example of function call
#gen_map('abstracts/','abstracts','BNB','Behavioral Domain')


