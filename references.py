'''
Created on Mar 11, 2016

@author: Jason
'''

import re
import pandas as pd
import os
import sys
import time
import signal
import numpy as np

newLinePattern = re.compile("[\\n\\r]+")


class ReferenceEntry(object):
    def __init__(self, authors, year, title, names):
        global newLinePattern
        self.authors = re.sub(newLinePattern, " ", authors)
        self.authorsArray = []
        self.authorsLastNameArray = []
        self.year = year
        self.title = title
        self.names = names
        self.index = -1
        self.count = 1

    def citationRegEx(self):
        ''' in text citation generator'''
        pat = "[\\(;\\s]{,2}"
        for i, a in enumerate(self.authorsLastNameArray):
            pat+=a+",?\\s*"
            if i + 2 == len(self.authorsLastNameArray):
                pat += "(and|&)\\s*"
        pat+=self.year+"[;\\)]"
        return pat

    def referenceRegEx(self):
        ''' in text reference finder generator -- may be better to find the title '''
        pat = ""
        for i, a in enumerate(self.authorsLastNameArray):
            pat+=a+".{,10}"

        pat+= self.year+".{,4}"+self.title
        return pat

    def getAuthors(self):
        global newLinePattern
        authorStr = re.sub(newLinePattern, " ", self.authors)
        self.authorsArray = []

        authors = re.finditer(self.names, authorStr)
        for a in authors:
            if a is not None:
                #print a.group(2)
                self.authorsArray.append(a.group(2))
                self.authorsLastNameArray.append(a.group(3))

    def getTitleRegEx(self):
        return re.compile(self.title.lower(), re.MULTILINE)

    def toArray(self):
        str_name = str(self.index)
        while len(str_name) < 5:
            str_name = "0" + str_name
        str_name = "ref_"+str_name
        return [self.authors, self.year, self.title, str_name, self.count]


class ReferenceGaz(object):
    def __init__(self):
        self.refs = []
        self.index = 0

    def insert(self, authors, year, title, names):
        global newLinePattern
        title = re.sub(newLinePattern, " ", title)
        #print "testing insert"
        for r in self.refs:
            if r.title.lower() == title.lower():
                r.count += 1
                return

        ref = ReferenceEntry(authors, year, title, names)
        self.refs.append(ref)
        self.index += 1
        ref.index = self.index


class MyException(Exception):
    pass


def timeout(signum, frame):
    raise MyException


def getReferences(fileName, reference, datesReg, names, gaz):
    text = ""
    with open(fileName, 'r') as f:
        for l in f:
            text += l
    sys.stdout.flush()
    #print text
    text = text[len(text)*3/4:]
    dates = re.finditer(datesReg, text)
    sys.stdout.flush()
    previousEnd = 0
    alarmAmount = 1
    curr = -1
    prev = time.clock()
    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.alarm(alarmAmount)
        curr = time.clock()
        for match in dates:
            signal.alarm(0)
            if match is not None:
                #print match.group(), match.start(), match.end()
                matchRefs = re.finditer(reference, text[max(match.start() -100, previousEnd):match.end()+300])
                signal.alarm(alarmAmount)
                for matchRef in matchRefs:
                    signal.alarm(0)
                    if matchRef is not None:
                        gaz.insert(matchRef.group(1), matchRef.group(14), matchRef.group(15), names)
                        previousEnd = matchRef.end()
            signal.alarm(alarmAmount)
    except:
        pass
    finally:
        signal.alarm(0)
    sys.stdout.flush()


def generate_references_gazetteer(pmids, text_dir):
    name = "([A-Z][^\\s\\d\\;\\:0-9\\.\\,\\)\\(]{,12})"
    names = "(("+name+"((,\\s{1,4})"+name+"\\.?)?(\\s{,4}"+name+"\\.?)?)(,\\s{1,4}((&|AND)\\s+)?)?)"
    datePar = "(\\(?(\\d{4})[a-f]?\\)?[\\.:]?)"
    title = "\\s{,4}([A-Z][^\\.0-9?!]+)(?<!\\s[A-z0-9])[\\.?!]"
    reg = "("+names+"{1,5})\\s{1,4}"+datePar+title
    reference = re.compile(reg, re.MULTILINE|re.DOTALL)
    
    gaz = ReferenceGaz()
    for i, pmid in enumerate(pmids):
        file_ = os.path.join(text_dir, pmid+".txt")
        print("Article {0}: {1}".format(i, pmid))
        sys.stdout.flush()
        getReferences(file_, reference, re.compile(datePar), re.compile(names, re.MULTILINE|re.DOTALL), gaz)

    outputArr = []
    for r in gaz.refs:
        if r.count >= 10:
            outputArr.append(r.toArray())

    df = pd.DataFrame(data=outputArr, columns=["authors", "year", "title", "ref_id", "occurrences"])
    return df


def extract_references(pmids, gazetteer_file, count_file, text_dir):
    """
    Creates feature table for references feature from text.
    """
    # Read in features
    #gazetteer = read_gazetteer(gazetteer_file)
    df = pd.read_csv(gazetteer_file)
    gazetteer = df.to_dict()

    # Count
    result_array = np.zeros((len(pmids), df.shape[0]))
    for i, pmid in enumerate(pmids):
        text_file = os.path.join(text_dir, pmid+".txt")
        with open(text_file, "r") as fo:
            text = fo.read()

        for j, keyword in enumerate(gazetteer['authors']):
            # ~ and ` are substitutes for { and } respectively because format freaks out about them when they are not indices
            reg_str = '{0}[^0-9]~,5`({1})?[^0-9]~,5`({2})'.format(gazetteer['authors'][j], gazetteer['year'][j], re.escape(gazetteer['title'][j]))
            reg_str = reg_str.replace('~','{').replace('`','}')
            regex = re.compile(reg_str, re.MULTILINE)
            result_array[i, j] += len(re.findall(regex, text))

    # Normalize matrix
    result_array = result_array / result_array.sum(axis=1)[:, None]    
    
    # Create and save output
    count_df = pd.DataFrame(columns=gazetteer['ref_id'].values(), index=pmids,
                            data=result_array)
    count_df.index.name = "pmid"
    count_df = count_df.fillna(0)
    count_df.to_csv(count_file)