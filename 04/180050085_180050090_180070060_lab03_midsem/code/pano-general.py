import os, sys
import numpy as np 
import cv2 as cv


def compute_kps(imgs):
	gray_imgs = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imgs]
	
	orb = cv.ORB_create()
	bundle = [orb.detectAndCompute(img, None) for img in gray_imgs]
	desc = [kp[1] for kp in bundle]
	kps = [kp[0] for kp in bundle]
	return kps, desc


def draw_kps(imgs, kps):
	l = len(imgs)
	objs = [cv.drawKeypoints(imgs[i], kps[i], None, color = (0, 255, 0)) for i in range(l)]
	for i in range(l):
		cv.imshow("Key Points {}".format(i), objs[i])
	cv.waitKey(-1)
	cv.destroyAllWindows()


def compute_matches(desc):
	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = False)
	matches = [[bf.knnMatch(t, q, k = 2) for q in desc] for t in desc]
	
	l = len(desc)
	raw_matches = [[len(matches[t][q]) for q in range(l)] for t in range(l)]
	print("Raw Matches:\n", raw_matches)

	# ratio test
	for t in range(l):
		for j in range(l):
			good = []
			for m, n in matches[t][q]:
				if m.distance < 0.75 * n.distance:
					good.append(m)
			matches[t][q] = good

	new_matches = [[len(matches[t][q]) for q in range(l)] for t in range(l)]
	print("Refined Matches:\n", new_matches)	

	return matches, np.asarray(new_matches)


def compute_homography(kps, matches):
	H = []
	for t in range(matches):
		tmp_h = []		
		for q in range(matches[t]):
			src = np.float32([kps[t][q][m.queryIdx].pt for m in matches[t][q]])
			dst = np.float32([kps[t][q][m.trainIdx].pt for m in matches[t][q]])
			h, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
			tmp_h.append(h)
		H.append(tmp_h)
	return H


def multi_mosaic(imgs, ref_idx):
	kps, desc = compute_kps(imgs)
	draw_kps(imgs, kps)
	matches, num_matches = compute_matches(desc)
	H = compute_homography(kps, matches)

	warped = [ref_idx]
	pano = imgs[ref_idx]
	non_warped = filter(lambda x: x not in warped, [i for i in range(len(imgs))])
	print(warped, non_warped)

	while len(warped) != len(imgs):
		# final img with highest matches with warped
		num = [np.amax(num_matches[t, warped]) for t in non_warped]
		t = num.index(max(num))
		
		


if __name__ == "__main__":
	path = sys.argv[1]
	ref_idx = sys.argv[2] - 1

	files = os.listdir(path)
	imgs = [cv.imread(os.path.join(path, f), 1) for f in sorted(files)]
	ref_img = imgs[ref_idx]

	# change function defn
	pano = multi_mosaic(imgs, ref_idx)

	# save final pano first
	dest_dir = os.path.normpath(path).split(os.sep)[-1]
	cv.imwrite("../results/pano-general-results/{}.png".format(dest_dir), pano)
	cv.imshow("result", pano)
	cv.waitKey(-1)
	cv.destroyAllWindows()

