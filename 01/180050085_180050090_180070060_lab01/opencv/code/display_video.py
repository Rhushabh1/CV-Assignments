'''streaming webcam vid'''
import numpy as np
import cv2 as cv
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Video display")
    parser.add_argument('vid_loc', type = str, metavar = 'v', help = 'Path to video')
    args = parser.parse_args()
    cap = cv.VideoCapture(args.vid_loc)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        font = cv.FONT_HERSHEY_SIMPLEX
        
        # Our operations on the frame come here
        cv.rectangle(frame,(490,30),(600,5),(255,255,255),-1)
        cv.putText(frame,'Deep Matrix',(500, 20),font,0.5,(0, 0, 0),1) 
        # Display the resulting frame
        cv.namedWindow('original')
        cv.moveWindow('original',100,100)
        cv.imshow('original',frame)
        
        # grayscale version of frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
        cv.rectangle(gray,(490,30),(600,5),(255,255,255),-1)
        cv.putText(gray,'Deep Matrix',(500, 20),font,0.5,(0, 0, 0),1)
        cv.namedWindow('grayscale version') 
        cv.moveWindow('grayscale version',1000,100)
        cv.imshow('grayscale version',gray)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
