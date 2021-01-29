import argparse


def norm(v, p = 2):
	ans = 0
	for num in v:
		ans += abs(num) ** p
	return ans ** (1/p)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'P-norm of floating point numbers')
	parser.add_argument('float_nums', type = float, nargs = '+', metavar = 'f', help = 'A floating point number from the vector')
	parser.add_argument('--p', type = int, default = 2, help = 'Power p')
	args = parser.parse_args()

	vec = args.float_nums
	p = args.p

	ans = norm(vec, p)
	print("Norm of {a} is {b:.2f}".format(a = vec, b = ans))
