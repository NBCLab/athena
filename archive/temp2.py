nlist = []
for i in athena.label_df['Paradigm Class']:
	nlist.append(len(i))

m_max = min(nlist)

glist = []
for x in nlist:
	if x == m_max:
		glist.append(x)

print len(glist)
print len(nlist)

nlist=[]
for x in athena.label_df['Behavioral Domain']:
	nlist.append(frozenset(x))

set(nlist)
print len(unique(nlist))



def unique_everseen(iterable):
	seen = set()
	for element in iterable:
		if element not in seen:
			seen.add(element)
			yield element

#print np.mean(nlist)
#print max(nlist)

nlist=[]
for i in athena.test_data['Abstract Text']:
	for x in i.split():
		nlist.append(x)
