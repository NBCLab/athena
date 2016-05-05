"""
Created on Thu May  5 13:21:52 2016
Obtain abstracts for pmids.

AB = Abstract
PT = Article type
TI = Title
AU = Authors
TA = Abbreviated journal name
DP = Date of publication

Sometimes:
OT = Author keywords
MAJR = MeSH Major Topic
SH = MeSH Subheadings
MH = MeSH Terms
@author: salo
"""

import os
import csv
from Bio import Entrez
from Bio import Medline
import pandas as pd


Entrez.email = "tsalo006@fiu.edu"

TERM = '("fMRI" OR "functional magnetic resonance imaging" OR "functional MRI") AND ("journal article"[PT] OR "introductory journal article"[PT])'

FORMATS = ["journal article", "introductory journal article"]



h = Entrez.esearch(db='pubmed', retmax='2', term=TERM)
result = Entrez.read(h)
print("Total number of publications containing {0}: {1}".format(TERM, result["Count"]))
h_all = Entrez.esearch(db='pubmed', retmax=result['Count'], term=TERM, )
result_all = Entrez.read(h_all)
pmids = sorted(list(result_all['IdList']))

df = pd.DataFrame(columns=["PMID"], data=pmids)
df.to_csv("fmri_pmids.csv", index=False)

#for pmid in pmids:
#    try:
#        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
#        record = list(Medline.parse(h))[0]
#        abstract = record["AB"]
#    except:
#        print pmid
#        print "Failed."
