import sys
import numpy as np 
import cv2 as cv

# keypoints
imgpoints = []

# callback functions
def add_kps(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN:
		imgpoints.append([x, y])


if __name__ == "__main__":
	try:
		path = "../data/calib_images/{}.jpg".format(sys.argv[1])
	except:
		path = "../data/calib_images/cube1.jpg"
	img = cv.imread(path, 1)

	# take imgpoints manually
	cv.namedWindow("Calibrate", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
	cv.resizeWindow("Calibrate", 1000, 750)
	cv.setMouseCallback("Calibrate", add_kps)
	cv.imshow("Calibrate", img)
	cv.waitKey(-1)
	cv.destroyAllWindows()

	# object points in 3D real world coordinate 
	# 23mm for 3x3 rubix cube
	# 15mm for chess board
	objp = np.zeros((6, 3), np.float32)
	objp[:,:2] = np.mgrid[0:3,0:2].T.reshape(-1,2)
	objpoints = [objp * 23]
	imgpoints = np.array([imgpoints]).astype(np.float32)

	# image and camera sensor specs
	gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	imageSize = gray_img.shape[::-1]
	apertureWidth, apertureHeight = 4.74/1.79, 4.74/1.79

	# compute camera matrix and focal length
	retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageSize, None, None)
	fovx, fovy, focalLength, principalPoint, aspectRatio = cv.calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight)

	print("Focal Length: {}".format(focalLength))