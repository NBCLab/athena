import pickle

for i in range(0,6):
	vocab = athena.pipeline[i].steps[0][1].vocabulary_
	pickle.dump(vocab,open('results/vocab_'+str(i)+'.p','wb'))


dim = athena.label_dimension_dict
dim_b = athena.dimension_beg
dim_e = athena.dimension_end
pickle.dump(dim  ,open('results/label_dimension_dict.p','wb'))
pickle.dump(dim_b,open('results/dim_beg.p','wb'))
pickle.dump(dim_e,open('results/dim_end.p','wb'))