import numpy as np 
import cv2 as cv

# keypoints
imgpoints = []

# callback functions
def add_kps(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN:
		imgpoints.append([x, y])


if __name__ == "__main__":

	#images for calibration
	img1 = cv.imread("../data/calib_images/cube1.jpg", 1)
	img2 = cv.imread("../data/calib_images/cube2.jpg",1)

	# take imgpoints manually
	cv.namedWindow("calibrate_img1", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
	cv.resizeWindow("calibrate_img1", 1000, 750)
	cv.setMouseCallback("calibrate_img1", add_kps)
	cv.imshow("calibrate_img1", img1)

	cv.waitKey(-1)
	cv.destroyAllWindows()

	cv.namedWindow("calibrate_img2", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
	cv.resizeWindow("calibrate_img2", 1000, 750)
	cv.setMouseCallback("calibrate_img2", add_kps)
	cv.imshow("calibrate_img2", img2)

	cv.waitKey(-1)
	cv.destroyAllWindows()

	# object points in 3D real world coordinate 
	# 23mm for 3x3 rubix cube
	# 15mm for chess board
	objp = np.zeros((6, 3), np.float32)
	objp[:,:2] = np.mgrid[0:3,0:2].T.reshape(-1,2)
	objpoints = [np.tile(objp * 23, (2,1))]
	imgpoints = np.array([imgpoints]).astype(np.float32)

	# image and camera sensor specs
	gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
	imageSize = gray_img.shape[::-1]
	apertureWidth, apertureHeight = 4.7/1.8, 4.7/1.8

	# compute camera matrix and focal length
	retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageSize, None, None)
	pixelSize = 0.0016
	fx = cameraMatrix[0, 0] * pixelSize
	fy = cameraMatrix[1, 1] * pixelSize
	focalLength = (fx + fy)/2

	print("Fx: {}mm\tFy: {}mm".format(round(fx, 5), round(fy, 5)))
	print("Focal Length: {0:.5f}mm".format(focalLength))