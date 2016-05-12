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
candidate_terms = ["subjects", "adults", "children", "users", "patients", "men", "women", "controls", "volunteers"]
cd_cand = re.compile(".*(\\b\d+)(.*[{0}]\\b).*".format("|".join(candidate_terms)), re.DOTALL)


def return_nes(sentences):
    nps = []
    sentences = parsetree(sentences)
    for sentence in sentences:
        for chunk in sentence.chunks:
            match = re.match(cd_np, str(chunk))
            if match is not None and any([term in nltk.word_tokenize(chunk.string) for term in candidate_terms]):
                np = match.group(1)
                nps += [np]
    nps = [(re.match(cd_cand, ne).group(2).strip(), re.match(cd_cand, ne).group(1).strip()) for ne in nps]
    return nps
                

Entrez.email = "tsalo006@fiu.edu"

TERM = ('("fMRI" OR "functional magnetic resonance imaging" OR "functional MRI") AND ' +
        '("journal article"[PT] OR "introductory journal article"[PT])')


h = Entrez.esearch(db="pubmed", retmax="2", term=TERM)
result = Entrez.read(h)
print("Total number of publications containing {0}: {1}".format(TERM, result["Count"]))
h_all = Entrez.esearch(db="pubmed", retmax=result["Count"], term=TERM, )
result_all = Entrez.read(h_all)
pmids = sorted(list(result_all["IdList"]))
pmids = [str(pmid) for pmid in pmids]

pmids = pmids[:500]

abstracts = [[] for _ in pmids]
samples = [[] for _ in pmids]
df = pd.DataFrame(columns=["PMID", "ABSTRACT", "SAMPLES"], data=map(list, zip(*[pmids, abstracts, samples])))
df.to_csv("fmri_pmids.csv", index=False)
df.set_index("PMID", inplace=True)

for pmid in df.index:
    try:
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = list(Medline.parse(h))[0]
    except:
        record = {}
        
    if "AB" in record.keys():
        df["ABSTRACT"].loc[pmid] = record["AB"]
    else:
        print pmid
        print "Failed."

for pmid in df.index:
    abstract = df["ABSTRACT"].loc[pmid]
    if abstract:
        sentences = nltk.sent_tokenize(abstract)
        sentences = [convert_words_to_numbers(sentence) for sentence in sentences]
        subj_sentences = find_candidates(sentences)
        num_sentences = " ".join(reduce_candidates(subj_sentences))
        nes = return_nes(num_sentences)
        if nes:
            df["SAMPLES"].loc[pmid] = nes
        else:
            df["SAMPLES"].loc[pmid] = "NONE FOUND"
df.to_excel("sample_sizes.xlsx")