import os, sys
import numpy as np 
import cv2 as cv
import math


def get_transformed_images(query_img, train_img):


	# # input point correspondences
	query_img_bw = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
	train_img_bw = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

	descriptor  = cv.ORB_create()
	kpq, ftq = descriptor.detectAndCompute(query_img_bw,None)
	kpt, ftt = descriptor.detectAndCompute(train_img_bw,None)


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
	x_max = int(math.ceil(np.max(modified_image_corners_X)))
	y_max = int(math.ceil(np.max(modified_image_corners_Y)))

	H_prime = np.eye(3)

	x_min = int(-math.floor(x_min))
	y_min = int(-math.floor(y_min))
	if(x_min >0):
		H_prime[0,2] = x_min
	else:
		x_min = 0
	if (y_min > 0):
		H_prime[1,2] = y_min
	else:
		y_min = 0
	print(H_prime)


	return H, x_min, y_min, x_max, y_max
		



if __name__ == "__main__":
	path = sys.argv[1]

	ref_idx = int(sys.argv[2]) - 1

	files = os.listdir(path)
	imgs = [cv.imread(os.path.join(path, f), 1) for f in sorted(files)]
	ref_img = imgs[ref_idx]
	other_images = imgs[:ref_idx] + imgs[ref_idx + 1:]

	# change function defn


	query_image = ref_img
	new_Hs = []
	max_x_deviation = 0
	max_y_deviation = 0
	max_x_co = ref_img.shape[1]
	max_y_co = ref_img.shape[0]
	for train_image in other_images:
		new_H, x_deviation, y_deviation,x_co, y_co = get_transformed_images(query_image, train_image)
		new_Hs.append(new_H)
		max_x_deviation = x_deviation if x_deviation > max_x_deviation else max_x_deviation
		max_y_deviation = y_deviation if y_deviation > max_y_deviation else max_y_deviation

		max_x_co = x_co if x_co > max_x_co else max_x_co
		max_y_co = y_co if y_co > max_y_co else max_y_co




	H_prime = np.eye(3)
	H_prime[0,2] = max_x_deviation
	H_prime[1,2] = max_y_deviation



	biggest_dimension = (max_x_co + max_x_deviation+1,max_y_co+ max_y_deviation+1, )



	for i in range(len(new_Hs)):
		H = H_prime @ new_Hs[i]
		train_img = other_images[i]
		pano = cv.warpPerspective(train_img, H, biggest_dimension)
		if i == 0:
			dst = pano
		else:
			mask = np.zeros(pano.shape)
			sum_pixels = pano.sum(axis = 2)

			mask = sum_pixels >0

			dst[mask] = pano[mask]

			

	pano = cv.warpPerspective(ref_img, H_prime, biggest_dimension)
	mask = np.zeros(pano.shape)
	sum_pixels = pano.sum(axis = 2)

	mask = sum_pixels >0

	dst[mask] = pano[mask]
	result = dst

	# save final pano first
	dest_dir = os.path.normpath(path).split(os.sep)[-1]
	cv.imwrite("../results/pano-general-results/{}.png".format(dest_dir), result)
	cv.imshow("result", result)
	cv.waitKey(-1)
	cv.destroyAllWindows()





