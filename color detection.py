#author@ Zhangjian Ouyang
#date 4/18/2018
import cv2
import numpy as np
import sys
import argparse
import imutils

reload(sys)
sys.setdefaultencoding('utf8')

cap = cv2.VideoCapture(0)

# set red thresh
lower_red=np.array([100, 40, 40])
upper_red=np.array([255, 1500, 150])

while(1):
    # get a frame and show
    ret, frame = cap.read()
    cv2.imshow('Capture', frame)

    # change to hsv model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get mask
    mask = cv2.inRange(hsv, lower_red, upper_red)
    cv2.imshow('Mask', mask)

    # detect blue
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('Result', res)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite("C:/personal files/appletest.jpg", res)

        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image")
        args = vars(ap.parse_args())

        # load the image, convert it to grayscale, blur it slightly,
        # and threshold it
        image = cv2.imread("C:/personal files/appletest.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        for c in cnts:
            # compute the center of the contour
            if cv2.contourArea(c) > 80:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # draw the contour and center of the shape on the image
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(image, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # show the image
        cv2.imshow("Image", image)
        print cX, cY

cap.release()
cv2.destroyAllWindows()
