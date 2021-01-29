import argparse
import numpy as np 


def gen_array(n):
	a = np.eye(n)
	odd = a[0::2, :]
	even = a[1::2, :]
	return np.concatenate((odd, even), axis = 0)

def crop_array(arr_2d:np.array, offset_height:int, offset_width:int, target_height:int, target_width:int):
	n = arr_2d[offset_height:offset_height + target_height, offset_width:offset_width + target_width]
	return n


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Matrix Row Manipulation')
	parser.add_argument('--N', type = int, default = 2, help = 'Matrix size')
	args = parser.parse_args()

	n = args.N

	orig_arr = gen_array(n)
	print("Original array:\n{}\n".format(orig_arr))

	i, r = n//2, n%2
	crop_arr = crop_array(orig_arr, i - 1, i - 1, i + r, i + r)
	print("Cropped array:\n{}\n".format(crop_arr))

	pad_arr = np.pad(crop_arr, (2, 2), mode = 'constant', constant_values = (0.5, 0.5))
	print("Padded array:\n{}\n".format(pad_arr))

	concat_arr = np.concatenate((pad_arr, pad_arr), axis = 1)
	print("Concatenated array: shape={a}\n{b}".format(a = concat_arr.shape, b = concat_arr))

