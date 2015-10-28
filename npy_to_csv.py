import numpy as np
import glob
from os import path

abs_dir = 'methods/heatmaps/*.npy'
save_dir = 'methods/csv/'

for filename in glob.glob(abs_dir):
	base = path.basename(filename)[:-8]
	tempf = np.load(filename)
	np.savetxt('/Users/dane/Desktop/heatmaps_oct_13/methods/csv/'+base+'.csv', tempf, delimiter=",")
	#np.savetxt(save_dir+base+'.csv', tempf, delimiter=",")