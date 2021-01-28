import numpy as np

import argparse
def crop_array(arr_2d:np.array, offset_height:int, offset_width:int, target_height:int, target_width:int):
    return arr_2d[offset_height:offset_height + target_height + 1, offset_width:offset_width + target_width + 1]



parser = argparse.ArgumentParser(description='Matrix Row Manipulation')

parser.add_argument('--N', type = int, help = 'Size of matrix')

args = parser.parse_args()

n = args.N

orig_mat = np.zeros((n,n))
ind_1 = n//2
if n %2 == 1:
    ind_1 +=1 
orig_mat[np.arange(0,ind_1), np.arange(0,n,2)] = 1
orig_mat[np.arange(ind_1, n), np.arange(1,n,2)] = 1
print("Original array:")

print(orig_mat)

print("\nCropped array:")
if n%2 == 0:
    cropped_array = crop_array(orig_mat, ind_1-1, ind_1-1, ind_1-1, ind_1-1)
else:
    cropped_array = crop_array(orig_mat, ind_1-2, ind_1-2, ind_1-1, ind_1-1)
    
print(cropped_array)

pad_array = np.pad(cropped_array, (2,2), 'constant', constant_values = (0.5,0.5))
print("\nPadded array:")
print(pad_array)
concat_arr = np.concatenate((pad_array, pad_array), axis = 1)
print(f"\nConcatenated array: shape={concat_arr.shape}")
print(concat_arr)
