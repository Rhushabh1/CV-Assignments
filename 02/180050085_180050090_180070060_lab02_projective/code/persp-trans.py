import cv2 as cv
import numpy as np
import argparse
if __name__ == '__main__':

    dist_image = cv.imread("../data/obelisk.png")
    imgray  = cv.cvtColor(dist_image, cv.COLOR_BGR2GRAY)
    threshold = 200
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv.contourArea)
    c = np.squeeze(c)
    # print(c.shape)
    left_sort = c[c[:,0].argsort()]
    top_sort = c[c[:,-1].argsort()]
    top_left = left_sort[0]
    top_right = top_sort[0]
    bottom_left = top_sort[-1]
    bottom_right = left_sort[-1]
    # print(top_left, top_right, bottom_left, bottom_right)

    # cv.drawContours(dist_image, [c], 0, (0,255,0), 3)
    old_corners = np.array([top_right,bottom_right,top_left, bottom_left])
    new_corners = np.array([[0,0], [511,0],[0,384],[511,384]])
    # print(old_corners.shape, new_corners.shape)
    M = cv.getPerspectiveTransform(np.float32(old_corners), np.float32(new_corners))
    # print(M.shape)
    identity_matrix = np.eye(M.shape[0])
    stencil = np.zeros(dist_image.shape).astype(dist_image.dtype)
    color = [255, 255, 255]
    cv.fillPoly(stencil, [c], color)
    result = cv.bitwise_and(dist_image, stencil)



    NUM_FRAMES = 1000
    for i in range(NUM_FRAMES+1):
        required_matrix = identity_matrix * (NUM_FRAMES - i)/NUM_FRAMES + M * i /(NUM_FRAMES)
        if i < 10:
            new_img = cv.warpPerspective(dist_image, required_matrix, (800,800))
        else:
            new_img = cv.warpPerspective(result, required_matrix, (800,800))
        cv.imshow('Warp:', new_img)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv.waitKey(-1)

    key = cv.waitKey(-1)
    cv.destroyAllWindows()
