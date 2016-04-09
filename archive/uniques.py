abs_list = []
for abs in athena.label_df['Abstract Text']:
	words = abs.split()
	abs_list.append(len(words))
	#words_set = set(words)
	#abs_list.append(len(words_set))