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

def load_calibration_matrix(file):
	final_path_folder = os.path.join(file, "calib")

'''Draw the front cover along with 
	the boundary of the implicit cuboid'''
def draw_book_on_img():
	pass

'''Add our own textures for the 
different planes of the book'''
def texture_book_on_img():
	pass

def calibrate_camera(file):
	objp = np.zeros((64,3), np.float32)
	objp[:,:2] = np.mgrid[0:8,0:8].T.reshape(-1,2)

	objpoints = []
	imgpoints = []

	for i in range(1,5):
		path = os.path.join(file, f"wall{i}.jpg")
		img = cv.imread(path,1)

		

		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

		ret, corners = cv.findChessboardCorners(gray, (8,8), None)

		if ret == True:
			objpoints.append(objp)
			corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

			imgpoints.apppend(corners)

			cv.drawChessboardCorners(img, (8,8), corners2, ret)
			cv.imshow('img', img)
			cv.waitKey(500)
	
	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	final_folderpath = os.path.join(file, "calib")

	np.save(os.path.join(final_folderpath,"ret.npy"), ret)
	np.save(os.path.join(final_folderpath,"mtx.npy"), mtx)
	np.save(os.path.join(final_folderpath,"dist.npy"), dist)
	np.save(os.path.join(final_folderpath,"rvecs.npy"), rvecs)
	np.save(os.path.join(final_folderpath,"tvecs.npy"), tvecs)

	return ret, mtx, dist, rvecs, tvecs






if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Introducing AR')
	parser.add_argument('file', metavar='F', help='Path to the AR data')
	parser.add_argument('-l', metavar='L', type=float, help='Length')
	parser.add_argument('-w', metavar='W', type=float, help='Width')
	parser.add_argument('-b', metavar='B', type=float, help='Breadth')
	parser.add_argument('-x', metavar='X', default=0, type=float, help='X-coordinate of book in CCS')
	parser.add_argument('-y', metavar='Y', default = 0,type=float, help='Y-coordinate of book in CCS')
	parser.add_argument('-theta', metavar='THETA', default = 0,type=float, help='Angle of inclination of the book')
	args = parser.parse_args()

	file = args.file
	L = args.l
	B = args.b
	W = args.w
	B = args.b
	X, Y = args.x, args.y
	theta = args.theta

	objp = np.zeros((49,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

	objpoints = []
	imgpoints = []

	for i in range(1,4):
		path = os.path.join(file, f"wall{i}.jpg")

		img = cv.imread(path,1)

		
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

		ret, corners = cv.findChessboardCorners(gray, (7,7), None)
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		print(ret)

		if ret == True:
			objpoints.append(objp)
			corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

			imgpoints.apppend(corners)

			cv.drawChessboardCorners(img, (7,7), corners2, ret)
			cv.imshow('img', img)
			cv.waitKey(-1)
	
	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	final_folderpath = os.path.join(file, "calib")

	np.save(os.path.join(final_folderpath,"ret.npy"), ret)
	np.save(os.path.join(final_folderpath,"mtx.npy"), mtx)
	np.save(os.path.join(final_folderpath,"dist.npy"), dist)
	np.save(os.path.join(final_folderpath,"rvecs.npy"), rvecs)
	np.save(os.path.join(final_folderpath,"tvecs.npy"), tvecs)

	cv.destroyAllWindows()
	# front_img, side_img, walls = get_data(file)
	
	# # set object points in WCS
	# objp = np.zeros((3*3, 3), np.float32)
	# objp[:, :2] = np.mgrid[0:3, 0:3].T.reshape(-1, 2)
	# axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
	# # book = np.float32([])

	# # criteria for cv2.cornerSubPix() function
	# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# for wall in walls:
	# 	gray_wall = cv.cvtColor(wall, cv.COLOR_BGR2GRAY)
	# 	cv.imshow("gray", gray_wall)
	# 	cv.waitKey(-1)
	# 	cv.destroyAllWindows()
	# 	# find initialised corners
	# 	ret, corners = cv.findCirclesGrid(gray_wall, (3, 3), None)
	# 	print(ret, corners)
	# 	# if ret:
	# # 		# refine corners accurately
	# # 		new_corners = cv.cornerSubPix(gray_wall, corners, (11, 11), (-1, -1), criteria)
	# # 		# find the rotation and translation vectors
	# # 		ret, rvecs, tvecs = cv.SolvePnP(objpoints, new_corners, cameraMatrix, distCoeffs)
	# # 		# project #D points onto image
	# # 		imgpoints, jacob = cv.projectPoints(book, rvecs, tvecs, cameraMatrix, distCoeffs)

	# # 		AR_img = draw_book_on_img(wall, new_corners, imgpoints)
	# # 		cv.namedWindow("Wall", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
	# # 		cv.resizeWindow("Wall", 1000, 750)
	# # 		cv.imshow("Wall", AR_img)