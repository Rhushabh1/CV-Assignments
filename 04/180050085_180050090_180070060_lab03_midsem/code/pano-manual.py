import os, sys
import numpy as np 
import cv2 as cv

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
	train_img, query_img = [cv.imread(os.path.join(path, f), 1) for f in sorted(files)]
	# query_img, train_img = [cv.imread(os.path.join(path, f), 1) for f in sorted(files)]

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

	h1, w1 = query_img.shape[:2]



	print(H.shape)

	# draw warped image
	pano = cv.warpPerspective(train_img, H, (train_img.shape[1] + query_img.shape[1], train_img.shape[0] + query_img.shape[0]))
	pano[0:query_img.shape[0], 0:query_img.shape[1]] = query_img
	
	# save final pano first
	dest_dir = os.path.normpath(path).split(os.sep)[-1]
	cv.imwrite("../results/pano-manual-results/{}.png".format(dest_dir), pano)
	cv.imshow("result", pano)
	cv.waitKey(-1)
	cv.destroyAllWindows()