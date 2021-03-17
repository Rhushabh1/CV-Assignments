import numpy as np

import cv2 as cv
import argparse

from numpy.core.defchararray import array, asarray
from numpy.core.numeric import array_equal


def order_of_coordinates(image):



    sort_by_y_coordinate = image[image[:,1].argsort()]
    final_res = np.zeros(image.shape)
    first_three_points = sort_by_y_coordinate[:3]
    last_three_points = sort_by_y_coordinate[3:]
    final_res[:3] = first_three_points[first_three_points[:,0].argsort()]
    final_res[3:] = last_three_points[last_three_points[:,0].argsort()]

    return final_res





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtain camera specification through files")

    parser.add_argument("-f", metavar= 'f', type = str, help = 'File path')

    args = parser.parse_args()

    file_path = args.f

    both_images = np.loadtxt(file_path).astype(np.int32)



    image_1 = both_images[:6]
    image_2 = both_images[6:]
    final_image_1 = order_of_coordinates(image_1)
    final_image_2 = order_of_coordinates(image_2)



    # objp1 = np.asarray([[1,0,0],[0,0,0],[0,1,0],[2,0,0],[1,1,0],[2,1,0]]).astype(np.float32)
    # objp2 = np.asarray([[2,1,0],[1,1,0],[0,0,0],[1,0,0],[0,1,0],[2,0,0]]).astype(np.float32)


    objp = np.asarray([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0]])
    imgpoints = np.asarray([final_image_1,final_image_2]).astype(np.float32)
    objpoints = [objp, objp]
    objpoints = np.asarray(objpoints).astype(np.float32)


    maxX = int(np.max(both_images[:,0]))
    maxY = int(np.max(both_images[:,1]))

    minloss = 10000
    minlossshape = (0,0)


    ## After some experimentation we got good results at i = 720, j = 1280

    for i in range(720,721):
        for j in range(1280,1281):
            imageSize = (i,j)
            loss = 0

            retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imageSize, None, None)

            # objp = np.zeros((6, 3), np.float32)
            # objp[:,:2] = np.mgrid[0:3,0:2].T.reshape(-1,2)
            # test_objpoints = objp * 23
            
            
            rep_imgpoints, jacob = cv.projectPoints(objpoints[0], rvecs[0], tvecs[0], cameraMatrix, distCoeffs)
            rep_imgpoints = np.squeeze(rep_imgpoints)
            test_imgpoints = final_image_1

            loss += np.sum(np.sqrt((rep_imgpoints - test_imgpoints)**2))



            rep_imgpoints, jacob = cv.projectPoints(objpoints[1], rvecs[1], tvecs[1], cameraMatrix, distCoeffs)
            rep_imgpoints = np.squeeze(rep_imgpoints)
            test_imgpoints = final_image_2

            loss += np.sum(np.sqrt((rep_imgpoints - test_imgpoints)**2))
            loss/= 12

            if loss <= minloss:
                minloss = loss
                minlossshape = imageSize
    

    print(minloss, imageSize)
            


