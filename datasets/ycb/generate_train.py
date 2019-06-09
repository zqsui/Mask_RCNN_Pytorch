# new_lines = []
# with open('train.txt', 'r') as train_f:
# 	lines = train_f.readlines()
# 	for line in lines:
# 		if not "syn" in line:
# 			new_lines.append(line)


# with open('new_train.txt', 'w') as train_w:
# 	for line in new_lines:
# 		train_w.write(line)	


with open('train.txt', 'w') as train_w:
	for i in xrange(1, 762):
		if i % 10 != 0:
			path = "data/0000/{:06}".format(i)
			train_w.write(path + '\n')


with open('test.txt', 'w') as test_w:
	for i in xrange(1, 762):
		if i % 10 == 0:
			path = "data/0000/{:06}".format(i)
			test_w.write(path + '\n')