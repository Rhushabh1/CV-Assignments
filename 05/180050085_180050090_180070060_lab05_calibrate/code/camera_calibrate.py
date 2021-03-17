import sys
import numpy as np 
import cv2 as cv

img = None

# keypoints
imgpoints = []
test_imgpoints = []

# callback functions
def mark(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),25,(0,255,0),8)
        test_imgpoints.append([x, y])

def add_kps(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN:
		imgpoints.append([x, y])

if __name__ == "__main__":
    ind = 0
    try:
        path = "../data/calib_images/cube{}.jpg".format(sys.argv[1])
        ind = int(sys.argv[1])-1
    except:
        path = "../data/calib_images/cube1.jpg"
    
    #test image
    img = cv.imread(path)
    cv.namedWindow("mark_test_img", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.resizeWindow("mark_test_img", 1000, 750)
    cv.setMouseCallback("mark_test_img", mark)
    cv.imshow("mark_test_img", img)

    cv.waitKey(-1)
    cv.destroyAllWindows()

    cv.imwrite("../data/true.jpg",img)

    #images for calibration
    img1 = cv.imread("../data/calib_images/cube1.jpg", 1)
    img2 = cv.imread("../data/calib_images/cube2.jpg",1)
    img3 = cv.imread("../data/calib_images/cube3.jpg",1)

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

    # object points in 3D real world coordinate 
    # 23mm for 3x3 rubix cube
    intake = 6
    objp = np.zeros((intake, 3), np.float32)
    objp[:,:2] = np.mgrid[0:intake/2,0:2].T.reshape(-1,2)
    objpoints = np.tile(objp * 23, (3, 1, 1))
    imgpoints = np.array(imgpoints).astype(np.float32).reshape(3, intake, 2)
    # print(objpoints.shape)
    # print(imgpoints.shape)

    # image and camera sensor specs
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imageSize = gray_img.shape[::-1]

    # compute camera matrix and focal length
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageSize, None, None)
    objp = np.zeros((6, 3), np.float32)
    objp[:,:2] = np.mgrid[0:3,0:2].T.reshape(-1,2)
    test_objpoints = objp * 23
    rep_imgpoints, jacob = cv.projectPoints(test_objpoints, rvecs[ind], tvecs[ind], cameraMatrix, distCoeffs)

    rep_imgpoints = rep_imgpoints.reshape((6,2)).astype(int)
    test_imgpoints = np.array(test_imgpoints)
    rep_err = 0
    for i in range(6):
        rep_err += np.sqrt(np.sum(np.square(rep_imgpoints[i, :] - test_imgpoints[i, :])))
    rep_err /= 6

    print("Reprojection Error: {}".format(rep_err))

    for i in range(6):
        cv.putText(img, 'X', tuple(rep_imgpoints[i]), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8, cv.LINE_AA) 

    cv.imwrite('../data/projected.jpg',img)
