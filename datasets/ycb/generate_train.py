new_lines = []
with open('train.txt', 'r') as train_f:
	lines = train_f.readlines()
	for line in lines:
		if not "syn" in line:
			new_lines.append(line)


with open('new_train.txt', 'w') as train_w:
	for line in new_lines:
		train_w.write(line)	
