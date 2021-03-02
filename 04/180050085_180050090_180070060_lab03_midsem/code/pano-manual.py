import os, sys
import numpy as np 
import cv2 as cv
import math
# keypoints
kpT = []
kpQ = []

# callback functions
def add_train_kps(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN:
		kpT.append(cv.KeyPoint(x, y, 5))

def add_query_kps(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN:
		kpQ.append(cv.KeyPoint(x, y, 5))


if __name__ == "__main__":
	path = sys.argv[1]
	files = os.listdir(path)

	# train img is the one to be transformed
	# train_img, query_img = [cv.imread(os.path.join(path, f), 1) for f in sorted(files)]
	query_img, train_img = [cv.imread(os.path.join(path, f), 1) for f in files]

	# input point correspondences
	cv.namedWindow("train", cv.WINDOW_NORMAL)
	cv.namedWindow("query", cv.WINDOW_NORMAL)

	cv.setMouseCallback("train", add_train_kps)
	cv.setMouseCallback("query", add_query_kps)

	# cv.resizeWindow("train", 500, 500)
	# cv.resizeWindow("query", 500, 500)

	cv.imshow("train", train_img)
	cv.imshow("query", query_img)
	
	cv.moveWindow("query", 1000, 0)
	cv.waitKey(-1)
	cv.destroyAllWindows()

	# draw matches
	matches = [cv.DMatch(i, i, 0) for i in range(min(len(kpT), len(kpQ)))]
	q = cv.drawMatches(train_img, kpT, query_img, kpQ, np.random.choice(matches, 100), None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv.imshow("matches", q)
	cv.waitKey(-1)
	cv.destroyAllWindows()

	# find homography
	if len(matches) >= 4:
		src = np.float32([kpT[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
		dst = np.float32([kpQ[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
		H, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
	else:
		raise AssertionError("Insufficient Keypoints (<4)")
	# print(H)

	image_corners = np.float32(
		[[[0,0]],[[train_img.shape[1]-1,0]], [[0,train_img.shape[0]-1]],[[train_img.shape[1]-1,train_img.shape[0]-1]]]
	)

	modified_image_corners = cv.perspectiveTransform(image_corners, H)
	modified_image_corners_X = modified_image_corners[:,:,0]
	modified_image_corners_Y = modified_image_corners[:,:,1]
	print(modified_image_corners)
	print(modified_image_corners_X, modified_image_corners_Y)

	x_min = np.min(modified_image_corners_X)
	y_min = np.min(modified_image_corners_Y)
	H_prime = np.eye(3)

	x_min = math.floor(x_min)
	y_min = math.floor(y_min)
	if(x_min <0):
		H_prime[0,2] = -x_min
	else:
		x_min = 0
	if (y_min < 0):
		H_prime[1,2] = -y_min
	else:
		y_min = 0
	
	print(H_prime)


	print(H)



	x_min = int(H_prime[0,2])
	y_min = int(H_prime[1,2])
	pano = cv.warpPerspective(train_img, H_prime @ H, (train_img.shape[1] + query_img.shape[1], train_img.shape[0] + query_img.shape[0]))
	pano[y_min:y_min + query_img.shape[0], x_min:x_min + query_img.shape[1]] = query_img
	cv.imshow("result", pano)
	cv.waitKey(-1)
	cv.destroyAllWindows()