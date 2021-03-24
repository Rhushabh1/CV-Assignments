import numpy as np 
import cv2 as cv 
import argparse
import os


'''Read the input data from the file'''
def get_data(file):
	front_img = cv.imread(os.path.join(file, "front.png"), 1)
	side_img = cv.imread(os.path.join(file, "side.png"), 1)
	walls = [cv.imread(os.path.join(file, "wall{}.jpg".format(i)), 1) for i in range(1,3)]
	return front_img, side_img, walls

'''Draw the front cover along with 
	the boundary of the implicit cuboid'''
def draw_book_on_img():
	pass

'''Add our own textures for the 
different planes of the book'''
def texture_book_on_img():
	pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Introducing AR')
	parser.add_argument('file', metavar='F', help='Path to the AR data')
	parser.add_argument('-w', metavar='W', type=float, nargs=3, help='Dimensions of the book in WCS')
	parser.add_argument('-b', metavar='B', type=float, nargs=1, help='Number of walls')
	parser.add_argument('-x', metavar='X', type=float, nargs=1, help='X-coordinate of book in CCS')
	parser.add_argument('-y', metavar='Y', type=float, nargs=1, help='Y-coordinate of book in CCS')
	parser.add_argument('-theta', metavar='THETA', type=float, nargs=1, help='Angle of inclination of the book')
	args = parser.parse_args()

	file = args.file
	W = args.w
	B = args.b[0]
	X, Y = args.x[0], args.y[0]
	theta = args.theta[0]

	front_img, side_img, walls = get_data(file)
	
	# set object points in WCS
	objp = np.zeros((3*3, 3), np.float32)
	objp[:, :2] = np.mgrid[0:3, 0:3].T.reshape(-1, 2)
	axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
	# book = np.float32([])

	# criteria for cv2.cornerSubPix() function
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	for wall in walls:
		gray_wall = cv.cvtColor(wall, cv.COLOR_BGR2GRAY)
		cv.imshow("gray", gray_wall)
		cv.waitKey(-1)
		cv.destroyAllWindows()
		# find initialised corners
		ret, corners = cv.findCirclesGrid(gray_wall, (3, 3), None)
		print(ret, corners)
		# if ret:
	# 		# refine corners accurately
	# 		new_corners = cv.cornerSubPix(gray_wall, corners, (11, 11), (-1, -1), criteria)
	# 		# find the rotation and translation vectors
	# 		ret, rvecs, tvecs = cv.SolvePnP(objpoints, new_corners, cameraMatrix, distCoeffs)
	# 		# project #D points onto image
	# 		imgpoints, jacob = cv.projectPoints(book, rvecs, tvecs, cameraMatrix, distCoeffs)

	# 		AR_img = draw_book_on_img(wall, new_corners, imgpoints)
	# 		cv.namedWindow("Wall", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
	# 		cv.resizeWindow("Wall", 1000, 750)
	# 		cv.imshow("Wall", AR_img)