import glob
from os import path

#Combines methods and abstracts
abs_dir = 'data/abstracts/'
methods_dir = 'data/methods/*.txt'
combine_dir = 'data/combined/'

abs_ext = '_a_p.txt'

for filename in glob.glob(methods_dir):
	text = None
	text2 = None

	f = open(filename)
	text = f.read()
	f.close()
	base = path.basename(filename)[:-4]

	f = open(abs_dir+base+abs_ext)
	text2 = f.read()
	f.close()

	combined_text = text + ' ' + text2

	f = open(combine_dir+base+'.txt', 'w')
	f.write(combined_text)
	f.close()
