import cv2 as cv
import numpy as np
import argparse
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Affine Transformation')
    # parser.add_argument("-mat", type = str, help = 'Choose between manual vs api')
    # args = parser.parse_args()

    # type_needed = args.mat
    # if type_needed not in ['manual', 'api']:
    #     raise Exception("Incorrect argument: choose manual or api")



    # parser = argparse

    dist_image = cv.imread("../data/obelisk.png")
    imgray  = cv.cvtColor(dist_image, cv.COLOR_BGR2GRAY)
    threshold = 200
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv.contourArea)
    c = np.squeeze(c)
    print(c.shape)
    left_sort = c[c[:,0].argsort()]
    top_sort = c[c[:,-1].argsort()]
    top_left = left_sort[0]
    top_right = top_sort[0]
    bottom_left = top_sort[-1]
    bottom_right = left_sort[-1]
    print(top_left, top_right, bottom_left, bottom_right)

    # cv.drawContours(dist_image, [c], 0, (0,255,0), 3)
    old_corners = np.array([top_right,bottom_right,top_left, bottom_left])
    new_corners = np.array([[0,0], [511,0],[0,384],[511,384]])
    print(old_corners.shape, new_corners.shape)
    M = cv.getPerspectiveTransform(np.float32(old_corners), np.float32(new_corners))

    new_img = cv.warpPerspective(dist_image, M, (512,385))

    cv.imshow('Warp:', new_img)
    cv.waitKey(0)