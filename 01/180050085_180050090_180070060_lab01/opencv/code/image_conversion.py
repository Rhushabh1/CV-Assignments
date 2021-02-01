import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Image Conversion')
	parser.add_argument('path', help = 'Path to image file')
	args = parser.parse_args()

	path = args.path

	orig_img = cv.imread(path, 1)
	norm_img = orig_img / 255
	inv_img = cv.cvtColor(orig_img, cv.COLOR_RGB2BGR)

	# matplotlib
	fig, (ax1, ax2) = plt.subplots(1, 2)
	fig.suptitle("Plotting images using matplotlib")
	ax1.imshow(inv_img)
	ax1.title.set_text("Original Image")
	ax2.imshow(inv_img / 255)
	ax2.title.set_text("Normalized Image")
	plt.savefig("../results/image_conversion_matplotlib.png")
	plt.show()

	# opencv
	norm_img = norm_img.astype(np.uint8)
	cv.imshow("Original Image", orig_img)
	cv.imshow("Normalized Image", inv_img)
	cv.imwrite("../results/image_conversion_opencv_orig.png", orig_img)
	cv.imwrite("../results/image_conversion_opencv_norm.png", inv_img)
	cv.waitKey(0)
	cv.destroyAllWindows()