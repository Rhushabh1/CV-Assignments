import argparse
import numpy as np 
import cv2 as cv 
import os


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Image Conversion')
	parser.add_argument('path', help = 'Path to image file')
	args = parser.parse_args()

	path = args.path
	num_imgs = 6

	imgs = [cv.imread(os.path.join(path,"display0"+str(i)+".png"), 1) for i in range(num_imgs)]
	idx = 0
	cv.imshow("Display Images", imgs[idx])

	key = cv.waitKey(0)
	while key in [ord('n'), ord('p')]:
		if key == ord('n'):
			idx = (idx + 1)%num_imgs
		else:
			idx = (idx - 1)%num_imgs

		cv.imshow("Display Images", imgs[idx])
		key = cv.waitKey(0)

	cv.destroyAllWindows()
