import argparse
import numpy as np 
import cv2 as cv 


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Image Conversion')
	parser.add_argument('path', help = 'Path to image file')
	args = parser.parse_args()

	path = args.path

	imgs = [cv.imread(path+"/display0"+i+".png", 1) for i in range(6)]
	cv.imshow(imgs[0])

	key = cv.waitKey(0)
	while key in [ord('n'), ord('p')]:
		print("pressed {}".format(key))
	print("wrong key")
