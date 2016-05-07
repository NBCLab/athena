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

from Bio import Entrez
from Bio import Medline
import pandas as pd
import nltk
from find_subjects import convert_words_to_numbers, find_candidates, reduce_candidates
from pattern.en import parsetree


Entrez.email = "tsalo006@fiu.edu"

TERM = '("fMRI" OR "functional magnetic resonance imaging" OR "functional MRI") AND ("journal article"[PT] OR "introductory journal article"[PT])'

FORMATS = ["journal article", "introductory journal article"]



h = Entrez.esearch(db="pubmed", retmax="2", term=TERM)
result = Entrez.read(h)
print("Total number of publications containing {0}: {1}".format(TERM, result["Count"]))
h_all = Entrez.esearch(db="pubmed", retmax=result["Count"], term=TERM, )
result_all = Entrez.read(h_all)
pmids = sorted(list(result_all["IdList"]))

df = pd.DataFrame(columns=["PMID"], data=pmids)
df.to_csv("fmri_pmids.csv", index=False)

pmids = pmids[:20]
abstracts = []
for pmid in pmids:
    try:
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
        abstracts += [record["AB"]]
    except:
        print pmid
        print "Failed."

abstracts2 = []
for abstract in abstracts:
    sentences = nltk.sent_tokenize(abstract)
    abstract2 = " ".join([convert_words_to_numbers(sentence) for sentence in sentences])
    abstracts2 += [abstract2]

for abstract in abstracts2:
    sentences = nltk.sent_tokenize(abstract)
    print len(sentences)
    sentences2 = find_candidates(sentences)
    print len(sentences2)
    sentences3 = reduce_candidates(sentences2)
    print len(sentences3)
    print sentences3

p = parsetree(sentences3[0], relations=True, lemmata=True)

for sentence in p:
    for chunk in sentence.chunks:
        print chunk.type, [(w.string, w.type) for w in chunk.words]
