
def conf(clf_name,label_dimension):
	label_pred = athena.estimators[clf_name]['label_pred']
	label_index_end = athena.dimension_end[label_dimension]
	label_index_beg = athena.dimension_beg[label_dimension]
	subset_true = athena.test_label[:, label_index_beg:label_index_end]
	subset_pred = label_pred[:, label_index_beg:label_index_end]
	conf_array = np.empty(shape=subset_true.shape)
	for (x,y), value in np.ndenumerate(subset_true):
		if subset_true[x,y] == 0 and subset_pred[x,y] == 0:
			conf_array[x,y] = 1
		elif subset_true[x,y] == 0 and subset_pred[x,y] == 1:
			conf_array[x,y] = 2
		elif subset_true[x,y] == 1 and subset_pred[x,y] == 0:
			conf_array[x,y] = 3
		elif subset_true[x,y] == 1 and subset_pred[x,y] == 1:
			conf_array[x,y] = 4
	lbls = list(athena.label_dimension_dict[label_dimension])
	np.save('results/heatmaps/'+clf_name+'_'+label_dimension+'.csv',conf_array, delimiter=',')
	f = open('results/heatmaps'+clf_name+'_'+label_dimension+'_label.txt', 'w')
	f.write(lbls)
	f.close()
	#scipy.io.savemat(clf_name+label_dimension+'.mat',mdict={(clf_name+label_dimension): conf_array})
	#scipy.io.savemat(clf_name+label_dimension+'labels.mat',mdict={'labels':lbls})
	'''
	from matplotlib import pyplot as plt
	column_labels = lbls
	heatmap = plt.pcolor(conf_array)
	plt.show()
	'''
	return conf_array


 
for col in ['Diagnosis','Stimulus Modality','Response Modality','Response Type','Stimulus Type', 'Instructions', 'Behavioral Domain', 'Paradigm Class']:
	for clf in ['MNB','LR','LRL1','SVC']:
		conf(clf,col)

x = conf('MNB','Diagnosis')


#257 test