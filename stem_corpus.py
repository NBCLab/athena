import glob
from nltk.stem.snowball import EnglishStemmer

abs_dir = 'data/abstracts/*.txt'
methods_dir = 'data/methods/*.txt'
combined_dir = 'data/combined/*.txt'

stemmer = EnglishStemmer()

def stem_corpus(text_directory):
	for filename in glob.glob(text_directory):
		f = open(filename)
		text = f.read()
		f.close()
		
		stem_list = []
		for word in text.split():
			stem_list.append(stemmer.stem(word))

		f = open('data/stemmed/'+filename[5:],'w+')
		for cword in stem_list:
			f.write(cword+' ')
		f.close()

stem_corpus(abs_dir)
stem_corpus(methods_dir)
stem_corpus(combined_dir)