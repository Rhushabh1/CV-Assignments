import numpy as np 
import cv2 as cv 
import argparse
import os


'''Read the input data from the file'''
def get_data(file):
	front_img = cv.imread(os.path.join(file, "front.png"), 1)
	side_img = cv.imread(os.path.join(file, "side.png"), 1)
	walls = [cv.imread(os.path.join(file, "wall{}.jpg".format(i)), 1) for i in range(1,4)]
	return front_img, side_img, walls

def load_calibration_matrix(file):
	final_path_folder = os.path.join(file, "calib")
	cameraMatrix = np.load(os.path.join(final_path_folder, "cameraMatrix.npy"))
	distCoeffs = np.load(os.path.join(final_path_folder, "distCoeffs.npy"))
	retval = np.load(os.path.join(final_path_folder, "retval.npy"))
	rvecs = np.load(os.path.join(final_path_folder, "rvecs.npy"))
	tvecs = np.load(os.path.join(final_path_folder, "tvecs.npy"))

	return cameraMatrix, distCoeffs, retval, rvecs, tvecs
'''Draw the front cover along with 
	the boundary of the implicit cuboid'''
def draw_book_on_img():
	pass

'''Add our own textures for the 
different planes of the book'''
def texture_book_on_img():
	pass

def draw_book_on_image(L,B,W, wall,rvec, tvec, cameraMatrix, distCoeffs, front_img, side_img):
	front_cover_positions = np.asarray([[0,B,W], [L,B,W], [L,0,W], [0,0,W]])*0.5

	back_cover_positions = np.asarray([[0,B,0], [L,B,0], [L,0,0], [0,0,0]])*0.5

	points_needed = np.vstack((front_cover_positions,back_cover_positions))

	rep_imgpoints, jacob = cv.projectPoints(points_needed, rvec, tvec, cameraMatrix, distCoeffs)

	rep_imgpoints = rep_imgpoints.squeeze()
	rep_imgpoints = np.around(rep_imgpoints).astype(np.int32)

	# Now warping the images


	new_front_cover = rep_imgpoints[:4]

	new_back_cover = rep_imgpoints[4:]
	side_cover = np.vstack((rep_imgpoints[4:6],rep_imgpoints[:2:-1]))


	img = wall


	img = cv.line(img, tuple(rep_imgpoints[0]), tuple(rep_imgpoints[1]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[1]), tuple(rep_imgpoints[2]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[2]), tuple(rep_imgpoints[3]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[3]), tuple(rep_imgpoints[0]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[4]), tuple(rep_imgpoints[5]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[5]), tuple(rep_imgpoints[6]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[6]), tuple(rep_imgpoints[7]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[7]), tuple(rep_imgpoints[4]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[0]), tuple(rep_imgpoints[4]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[1]), tuple(rep_imgpoints[5]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[2]), tuple(rep_imgpoints[6]),(0,0,255),4)
	img = cv.line(img, tuple(rep_imgpoints[3]), tuple(rep_imgpoints[7]),(0,0,255),4)

	## warp for front cover
	max_front_pix = front_img.shape[0] - 1
	src_frontcover = np.asarray([[0,max_front_pix],[max_front_pix, max_front_pix], [max_front_pix, 0], [0,0]]).astype(np.float32)

	
	print(src_frontcover.shape)
	M = cv.getPerspectiveTransform(src_frontcover, new_front_cover.astype(np.float32))

	final_image = cv.warpPerspective(front_img, M, tuple(img.shape[:2:-1]))


	mask = np.zeros(img.shape)

	sum_pixels = final_image.sum(axis = 2)

	mask = sum_pixels > 0

	img[mask] = final_image[mask]






	return img

	




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
	X, Y = args.x, args.y
	theta = args.theta

	cameraMatrix, distCoeffs, retval, rvecs, tvecs = load_calibration_matrix(file)
	front_img, side_img, walls = get_data(file)

	print(side_img.shape)


	for i in range(3):
		wall = walls[i]

		img = draw_book_on_image(L, B,W, wall,rvecs[i], tvecs[i], cameraMatrix, distCoeffs, front_img, side_img)

		height, width = img.shape[:2]
		cv.namedWindow('jpg', cv.WINDOW_NORMAL)

		cv.resizeWindow('jpg', int(width/4), int(height/4))
		cv.imshow("jpg", img)
		cv.waitKey(-1)

		cv.destroyAllWindows()

		

	



