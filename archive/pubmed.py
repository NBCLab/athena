import os
import time
import urllib2
import pandas as pd
import re

#http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=18056543&retmode=xml&rettype=abstract'
#Simple way to make a directory if it doesn't exist
def checkMakeDirectory(directory):
	if not os.path.exists(directory):
  		os.makedirs(directory)

##-------------------------------------------------------##
#                 getAbstract(int PMID)                   #
#---------------------------------------------------------#
# Return Values:  -1 failure,   0 success                 #
#---------------------------------------------------------#
# Queries pubmed by PMID for abstract in XML format       #
# Parses text and keeps text in <AbstractText> tag only   #
# Saves file under PMID_a.txt                             #
##-------------------------------------------------------##
def getAbstract(PMID,baseDirectory='./abstracts/'):
	baseURL   = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='
 	endURL    = '&retmode=xml&rettype=abstract'
 	#Construct URL to query pubmed
 	totalURL = baseURL+str(PMID)+endURL
 	try:
 		response = urllib2.urlopen(totalURL)
 		html = response.read()
 		#Response contains other info fields, grab only the abstract text
 		html = html.split('<Abstract>')[1].split('</Abstract>')[0]
 		#Use regular expression to get rid of unwanted symbols
 		html = re.sub('<[^>]+>', '', html)
 		#Check if dir exists, if not create it
 		checkMakeDirectory(baseDirectory)
 		#Write abstract to file
		f = open(baseDirectory+str(PMID)+'_a.txt', 'w')
		f.write(html)
		f.close()
		#Return 0 on success 
		return 0
	#Failure, most likely 404 page from invalid PMID
 	except Exception as e:
 		print('PMID '+str(PMID)+': 404')
 		return (-1)


def grabAllAbstracts():
	corpusDir = '../athena_rawdata/meta_data/'
	f = open('medaDataPMIDs.txt', 'w')
	for filename in os.listdir(corpusDir):
		df = pd.read_csv(corpusDir+filename)
		print filename
		PMIDs = df['PubMed ID']
		PMIDs = PMIDs.drop_duplicates()
		for pmid in PMIDs:
			f.write(str(pmid)+'\n')
			print (str(pmid))
			#getAbstract(pmid)
			#time.sleep(0.1)

grabAllAbstracts()
#getAbstract(18056543)