import shutil
import os
filename_pmid = './filename_pmid.txt'
methods_loc = './ALL/'

d = dict()
with open(filename_pmid) as fin:
    for line in fin:
        linelist = line.split()
        if len(linelist) > 1:
            filename = linelist[0][:-4]
            pubmedid = linelist[1]
            if 'sparse' in filename:
                filename = filename[:-7]
            d[filename] = pubmedid

for key in d.keys():
    old_file = '/Users/dane/Desktop/dAthena/ALL/'+key+'_p.txt'
    new_file = '/Users/dane/Desktop/dAthena/methods/'+d[key]+'.txt'
    os.system ("cp %s %s" % (old_file, new_file))
    #shutil.copy2(old_file, new_file)