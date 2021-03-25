import sys
import numpy as np 
import cv2 as cv

img = None

# keypoints
imgpoints = []

# callback functions
def add_kps(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN:
		imgpoints.append([x, y])

if __name__ == "__main__":
    
    #images for calibration
    img1 = cv.imread("../data/augment_book/wall1.jpg", 1)
    img2 = cv.imread("../data/augment_book/wall2.jpg",1)
    img3 = cv.imread("../data/augment_book/wall3.jpg",1)
    img4 = cv.imread("../data/augment_book/wall4.jpg",1)

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

    cv.namedWindow("calibrate_img3", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow("calibrate_img3", 1000, 750)
    cv.setMouseCallback("calibrate_img3", add_kps)
    cv.imshow("calibrate_img3", img3)

    cv.waitKey(-1)
    cv.destroyAllWindows()

    cv.namedWindow("calibrate_img4", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow("calibrate_img4", 1000, 750)
    cv.setMouseCallback("calibrate_img4", add_kps)
    cv.imshow("calibrate_img4", img4)

    cv.waitKey(-1)
    cv.destroyAllWindows()

    # object points in 3D real world coordinate 
    # 15mm for chess board
    intake = 4
    objp = np.zeros((intake, 3), np.float32)
    objp[:,:2] = np.mgrid[0:intake/2,0:2].T.reshape(-1,2)
    objpoints = np.tile(objp * 15, (4, 1, 1))
    imgpoints = np.array(imgpoints).astype(np.float32).reshape(4, intake, 2)

    # image and camera sensor specs
    gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    imageSize = gray_img.shape[::-1]

    # compute camera matrix and focal length
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageSize, None, None)
    
    np.save("../data/augment_book/calib/ret.npy", ret)
    np.save("../data/augment_book/calib/mtx.npy", mtx)
    np.save("../data/augment_book/calib/dist.npy", dist)
    np.save("../data/augment_book/calib/rvecs.npy", rvecs)
    np.save("../data/augment_book/calib/tvecs.npy", tvecs)

    