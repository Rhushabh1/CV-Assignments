import os, sys
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
import math
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
	query_img, train_img = [cv.imread(os.path.join(path, f), 1) for f in files]

	# # input point correspondences
	query_img_bw = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
	train_img_bw = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

	descriptor  = cv.ORB_create()
	kpq, ftq = descriptor.detectAndCompute(query_img_bw,None)
	kpt, ftt = descriptor.detectAndCompute(train_img_bw,None)


	fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
	ax1.imshow(cv.drawKeypoints(train_img,kpq,None,color=(0,255,0)))
	ax1.set_xlabel("(a)", fontsize=14)
	ax2.imshow(cv.drawKeypoints(query_img,kpt,None,color=(0,255,0)))
	ax2.set_xlabel("(b)", fontsize=14)
	plt.show()


	# create BFMatcher object
	# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
	# matches = bf.knnMatch(ftq, ftt, k = 2)
	# # ratio test
	# good = []
	# for m, n in matches:
	# 	if m.distance < 0.75 * n.distance:
	# 		good.append(m)
	# matches = good
	# # Draw first 10 matches.
	# img3 = cv.drawMatches(query_img,kpq,train_img,kpt,matches[:35],None, flags=2)


	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	matches = bf.match(ftq, ftt)

	img3 = cv.drawMatches(query_img,kpq,train_img,kpt,np.random.choice(matches, 35),None, flags=2)

	# matches = (sorted(matches, key = lambda d :d.distance,reverse = True))[:10]
	# im_matches = cv.drawMatches(train_img,kpt,query_img,kpq,matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	cv.imshow("matches", img3)
	cv.waitKey(0)
	cv.destroyAllWindows()


	kpT = np.float32([kp.pt for kp in kpt])
	kpQ = np.float32([kp.pt for kp in kpq])
	# get correspondences

	if len(matches) >= 4:
		src = np.float32([kpT[m.trainIdx] for m in matches]).reshape(-1,1,2)
		dst = np.float32([kpQ[m.queryIdx] for m in matches]).reshape(-1,1,2)
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
	
	# save final pano first
	dest_dir = os.path.normpath(path).split(os.sep)[-1]
	cv.imwrite("../results/pano-auto-results/{}.png".format(dest_dir), pano)
	cv.imshow("result", pano)
	cv.waitKey(0)
	cv.destroyAllWindows()