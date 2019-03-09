import sys

sum_matrix = 0
sum_diag = 0
idx = 0

has_comma = False
for line in sys.stdin:
	if "," in line:
		has_comma = True
	line_strip = line.strip().strip("[ ]")
	if has_comma:
		line_split = map(lambda x: x.split()[0], line_strip.split(","))
	else:
		line_split = line_strip.split()
	if len(line_split) > 0:
		num_class = len(line_split)
		int_list = map(int, map(float, line_split))
		sum_matrix += sum(int_list)
		sum_diag += int_list[idx]
		idx += 1

	if idx == num_class:
		break

print 1.*sum_diag/sum_matrix