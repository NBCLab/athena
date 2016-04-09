import numpy as np
import glob

#results_dir = 'methods/*.npy'
results_dir = 'abstracts/*.npy'

def combine(r_dir):
	r_ary = np.empty([4,9])
	f_list = []
	for filename in glob.glob(r_dir):
		f_list.append(np.load(filename))
	for arr in f_list:
		r_ary = np.add(r_ary,arr)
	r_ary = np.divide(r_ary,10)
	#np.savetxt("methods_f1.csv", r_ary, delimiter=",")
	np.savetxt("abstracts_f1.csv", r_ary, delimiter=",")
	return r_ary

combine(results_dir)
