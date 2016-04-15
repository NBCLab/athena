'''
Created on Mar 11, 2016

@author: Jason
'''
import re
import numpy as np
import pandas as pd
from os import listdir
from os.path import join, isfile
import sys, time
import signal
newLinePattern = re.compile("[\\n\\r]+")
class ReferenceEntry(object):
    def __init__(self, authors, year, title, names):
        global newLinePattern
        self.authors = re.sub(newLinePattern, " ", authors)
        self.authorsArray = []
        self.authorsLastNameArray = []
        self.year = year
        #self.title = re.sub(newLinePattern, " ", title)
        self.title = title
        self.names = names
        #self.getAuthors()
        #self.citationPattern = self.citationRegEx()
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
        return re.compile(title.toLower(), re.MULTILINE)
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
            if r.title.toLower() == title.toLower():
                r.count += 1
                return

        ref = ReferenceEntry(authors, year, title, names)
        self.refs.append(ref)
        self.index += 1
        ref.index = self.index

def create_gaz(article_root):
    name = "([A-Z][^\\s\\d\\;\\:0-9\\.\\,\\)\\(]{,12})"
    names = "(("+name+"((,\\s{1,4})"+name+"\\.?)?(\\s{,4}"+name+"\\.?)?)(,\\s{1,4}((&|AND)\\s+)?)?)"
    datePar = "(\\(?(\\d{4})[a-f]?\\)?[\\.:]?)"
    title = "\\s{,4}([A-Z][^\\.0-9?!]+)(?<!\\s[A-z0-9])[\\.?!]"
    #reg = "[\\n\\r]+("+names+"{1,6})\\s+"+datePar+title
    reg = "("+names+"{1,5})\\s{1,4}"+datePar+title
    reference = re.compile(reg, re.MULTILINE|re.DOTALL)
    
    articleList = [join(article_root, f) for f in listdir(article_root) if isfile(join(article_root, f)) and f[-3:] == "txt"]
    gaz = ReferenceGaz()
    print reg
    for i, a in enumerate(articleList):
        print 'article', i, a
        sys.stdout.flush()
        getReferences(a, reference, re.compile(datePar), re.compile(names, re.MULTILINE|re.DOTALL), gaz)
        print len(gaz.refs)

    outputArr = []

    for r in gaz.refs:
        outputArr.append(r.toArray())
    df = pd.DataFrame(outputArr)
    df.columns = ['authors', 'year', 'title', 'ref_id', 'occurrences']
    df.to_csv("references_gaz.csv")
class MyException(Exception):
    pass
def timeout(signum, frame):
    raise MyException
def getReferences(fileName, reference, datesReg, names, gaz):
    #print reg
    text = ""
    with open(fileName, 'r') as f:
        for l in f:
            text += l
    print "finding"
    sys.stdout.flush()
    #print text
    text = text[len(text)*3/4:]
    dates = re.finditer(datesReg, text)
    print "found"
    sys.stdout.flush()
    previousEnd = 0

    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.alarm(1)
        for match in dates:
            signal.alarm(0)
            if match is not None:
                #print match.group(), match.start(), match.end()
                matchRefs = re.finditer(reference, text[max(match.start() -100, previousEnd):match.end()+300])
                signal.alarm(1)
                for matchRef in matchRefs:
                    signal.alarm(0)
                    if matchRef is not None:
                        gaz.insert(matchRef.group(1), matchRef.group(14), matchRef.group(15), names)
                        wasChange = True
                        previousEnd = matchRef.end()
            signal.alarm(1)
    except:
        pass
    print "done"
    sys.stdout.flush()

if __name__ == "__main__":
    create_gaz("../ReferenceData")
