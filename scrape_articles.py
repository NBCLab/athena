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
import re


cd_np = re.compile("Chunk\('([^0-9]*(\d+)[^0-9]*)/NP", re.MULTILINE|re.DOTALL)
candidate_terms = ["subjects", "users", "patients", "men", "women", "male", "female", "controls"]
cd_cand = re.compile(".*(\\b\d+)(.*[{0}]\\b).*".format("|".join(candidate_terms)), re.DOTALL)


def return_nes(sentences):
    nps = []
    sentences = parsetree(sentences) 
    for sentence in sentences:
        for chunk in sentence.chunks:
            #print chunk.words
            match = re.match(cd_np, str(chunk))
            print chunk
            if match is not None and any([term in nltk.word_tokenize(chunk.string) for term in candidate_terms]):
                np = match.group(1)
                print np
                nps += [np]
    return nps
                

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
    sentences2 = find_candidates(sentences)
    sentences3 = reduce_candidates(sentences2)
    print sentences3
    nes = [return_nes(sentence) for sentence in sentences3]
    nes = [ne for ne in nes if ne]
    nes = [n for ne in nes for n in ne]
    nes = [(re.match(cd_cand, ne).group(2).strip(), re.match(cd_cand, ne).group(1).strip()) for ne in nes]
    print nes
