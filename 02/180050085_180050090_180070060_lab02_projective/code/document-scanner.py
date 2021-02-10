import numpy as np 
import argparse
import cv2 as cv 
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document Scanner')
    parser.add_argument("-i", type = str, help = 'path to document')
    args = parser.parse_args()

    h, w = 750, 600
    dist_doc = cv.imread(args.i)
    # dist_doc = cv.resize(dist_doc, (w, h))
    h, w = dist_doc.shape[0:2]

    imgray = cv.cvtColor(dist_doc, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv.contourArea)
    c = np.squeeze(c)

    # detect corners
    # distance from (0, 0)
    dist1 = c[:, 0]**2 + c[:, -1]**2
    # distance from (w-1, 0)
    dist2 = (c[:, 0] - w + 1)**2 + c[:, -1]**2

    x_sort = c[ dist1.argsort() ]
    y_sort = c[ dist2.argsort() ]
    top_left = x_sort[0]
    top_right = y_sort[0]
    bottom_left = y_sort[-1]
    bottom_right = x_sort[-1]

    # pt = np.array([bottom_left])
    # contour_img = cv.drawContours(dist_doc, [c], -1, (0, 255, 0), 5)
    # cv.imshow("Doc Contours", contour_img)
    # cv.waitKey(3)

    # perspective transform
    old_corners = np.array([top_left, top_right, bottom_left, bottom_right])
    new_corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    M = cv.getPerspectiveTransform(np.float32(old_corners), np.float32(new_corners))

    # show final doc
    final_doc = cv.warpPerspective(dist_doc, M, (w, h))
    # cv.imshow("Warped Document", final_doc)
    filename = os.path.basename(args.i)
    cv.imwrite("../results/"+filename, final_doc)

    key = cv.waitKey(-1)
    cv.destroyAllWindows()

