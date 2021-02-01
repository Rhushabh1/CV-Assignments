import argparse
import numpy as np 
import matplotlib.pyplot as plt


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Principal Component Analysis')
	parser.add_argument('path', help = 'Path to grid file')
	args = parser.parse_args()

	path = args.path
	X = np.loadtxt(path, dtype = 'i', delimiter = ',')

	n, d = X.shape
	X_new = (X - np.mean(X, axis = 0))/np.std(X, axis = 0, ddof = 1)
	cov = (X_new.T @ X_new)/n

	eig_val, eig_vec = np.linalg.eig(cov)
	idx = eig_val.argsort()[::-1]
	eig_val = eig_val[idx]
	eig_vec = eig_vec[:, idx]

	U = eig_vec[:, :2]
	Z = X_new @ U

	plt.scatter(Z[:, 0], Z[:, 1], color = 'b', s = 2)
	plt.title("PCA")
	plt.xlim([-15, 15])
	plt.ylim([-15, 15])
	plt.gca().set_aspect(1)
	# plt.show()
	plt.savefig("../results/out.png")
