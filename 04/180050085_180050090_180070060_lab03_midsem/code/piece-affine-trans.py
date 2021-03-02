import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
import cv2
from skimage import data

# image = data.astronaut()
image = cv2.imread('../data/piece/brick.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rows, cols = image.shape[0], image.shape[1]

for i in range(rows):
    if not np.array_equal(image[i,0],image[0,0]):
        y_axis = i
        break

fl = 0
for i in range(rows):
    for j in range(cols):
        if not np.array_equal(image[i,j],image[0,0]):
            y_high = i
            fl = 1
            break
    if fl:
        break        

amplitude = y_axis-y_high

src_cols = np.linspace(0, cols, 20)
src_rows = np.linspace(0, rows, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]

# add sinusoidal oscillation to row coordinates
dst_rows = src[:, 1] + np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * amplitude
dst_cols = src[:, 0]
dst = np.vstack([dst_cols, dst_rows]).T


tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

out_rows = image.shape[0] 
out_cols = cols
out = warp(image, tform, output_shape=(out_rows, out_cols))

for i in range(rows):
    if not np.array_equal(np.round(out[i,0],3),np.round(out[0,0],3)) and np.sum(out[i,0])!=0:
        start = i
        break

for i in range(rows-1,-1,-1):
    if not np.array_equal(np.round(out[i,0],3),np.round(out[0,0],3)) and np.sum(out[i,0])!=0:
        end = i
        break

fig, ax = plt.subplots()
ax.imshow(out[start+3:end+1,:,:])
plt.savefig("../results/piece-affine-results/brick.png")
plt.show()