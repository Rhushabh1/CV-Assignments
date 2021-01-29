import argparse
import numpy as np 


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Shape Manipulation')
	parser.add_argument('path', help = 'Path to grid file')
	args = parser.parse_args()

	path = args.path
	mat = np.loadtxt(path, dtype = 'i', delimiter = ',')

	m = int(input("M = "))
	n = int(input("N = "))

	new_mat = mat.reshape(mat.shape + (1, 1))
	new_mat = np.tile(new_mat, (1, 1, m, n))
	print(new_mat)