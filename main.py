import numpy as np
import cv2 as cv2
import glob
import mediapipe as mp
import cv2
import time


objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cap_file = cv2.VideoCapture(0)
success, image = cap_file.read()

while cap_file.isOpened():
    success, image = cap_file.read()
    if not success:
        print("empty camera frame")
        continue
    image = cv2.flip(image, 1)

    # image = cv2.imread('left01.jpg')
    # print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3,3),np.float32)/9
    gray = cv2.filter2D(gray,-1,kernel)
    cv2.imwrite('myleft.jpg', gray)

    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if ret == True:
        print("found")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(image, (7,6), corners2, ret)
        cv2.imshow('image', image)
        # cv2.waitKey(500)
    else:
        cv2.imshow('image', gray)
        # cv2.waitKey(500)


    if cv2.waitKey(5) & 0xFF == 27:
        break

# cap_file.release()
