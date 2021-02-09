import cv2
import numpy as np
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Affine Transformation')
    parser.add_argument("-mat", type = str, help = 'Choose between manual vs api')
    args = parser.parse_args()

    type_needed = args.mat
    if type_needed not in ['manual', 'api']:
        raise Exception("Incorrect argument: choose manual or api")



    parser = argparse

    dist_image = cv2.imread("../data/distorted.jpg")
    size_X, size_Y = dist_image.shape[0:2]
    size_X -= 1
    size_Y -= 1
    done = False
    print(dist_image[599,599])
    print(size_X, size_Y)
    x,y = -1, -1
    min_score = 300
    for i in range(size_X,-1,-1):
        for j in range(size_Y,-1,-1):
            if np.sum(dist_image[i,j]) > 300:
                print(np.sum(dist_image[i,j]))
                x,y = i,j
                done = True
                break
        if done:
            break
    y_2 = -1
    for j in range(size_Y):
        if np.sum(dist_image[599,j]) > 300:
            y_2 = j
            break

    x_2 = -1

    for i in range(size_X):
        if np.sum(dist_image[i, 599]) > 300:
            x_2 = i
            break

    
    if type_needed == 'manual':
        p = ((x - 599)/599 + x_2/599)/2
        q = ((y - 599)/599 + y_2/599)/2

        print(x - 599,x_2)
        print(y - 599, y_2)

        p = round(p,3)
        q = round(q,3)
        print(p,q)
        
        orig_transform = np.asarray([[1,p,0],[q,1,0]])
        # warp_mat = cv2.warpAffine(dist_image, orig_transform, (600,600))
        
        warp_mat = cv2.warpAffine(dist_image, orig_transform, (600,600), flags = cv2.WARP_INVERSE_MAP)
    else:
        
        source_points = np.array([[x,y],[x_2,599],[599,y_2]]).astype(np.float32)
        destination_points = np.array([[599,599], [0,599], [599,0]]).astype(np.float32)
        print(source_points.shape, destination_points.shape)
        M = cv2.getAffineTransform(source_points,destination_points)
        warp_mat = cv2.warpAffine(dist_image, M, (600,600))

        
    
    cv2.imshow('Warp:', warp_mat)
    cv2.waitKey(0)



