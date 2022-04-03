#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tkinter
from tkinter import messagebox
# import tkMessageBox

square_side_length = 23.0 # チェスボード内の正方形の1辺のサイズ(mm)
grid_intersection_size = (10, 7) # チェスボード内の格子数

pattern_points = np.zeros( (np.prod(grid_intersection_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(grid_intersection_size).T.reshape(-1, 2)
pattern_points *= square_side_length
object_points = []
image_points = []

root = tkinter.Tk()
root.withdraw()

video_input = cv2.VideoCapture(1)
if (video_input.isOpened() == False):
    exit()

camera_mat, dist_coef = [], []

if messagebox.askyesno('askyesno','キャリブレーションデータ(K.csv, d.csv)を読み込みますか？'):
    # キャリブレーションデータの読み込み
    camera_mat = np.loadtxt('K.csv', delimiter=',')
    dist_coef = np.loadtxt('d.csv', delimiter=',')
    print( "K = \n", camera_mat)
    print( "d = ", dist_coef.ravel())
else:
    # チェスボードの撮影
    capture_count = 0
    while(True):
        ret, frame = video_input.read()

        # チェスボード検出用にグレースケール画像へ変換
        #grayscale_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # チェスボードのコーナーを検出
        #found, corner = cv2.findChessboardCorners(grayscale_image, grid_intersection_size)
        found, corner = cv2.findChessboardCorners(frame, grid_intersection_size)

        if found == True:
            print( 'findChessboardCorners : True')

            # 現在のOpenCVではfindChessboardCorners()内で、cornerSubPix()相当の処理が実施されている？要確認
            #term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            #cv2.cornerSubPix(grayscale_image, corner, (5,5), (-1,-1), term)
            #cv2.drawChessboardCorners(grayscale_image, grid_intersection_size, corner, found)

            cv2.drawChessboardCorners(frame, grid_intersection_size, corner, found)
        if found == False:
            print( 'findChessboardCorners : False')

        cv2.putText(frame, "Enter:Capture Chessboard(" + str(capture_count) + ")", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        cv2.putText(frame, "N    :Completes Calibration Photographing", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        cv2.putText(frame, "ESC  :terminate program", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        #cv2.putText(grayscale_image, "Enter:Capture Chessboard(" + str(capture_count) + ")", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        #cv2.putText(grayscale_image, "ESC  :Completes Calibration Photographing.", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        cv2.imshow('original', frame)
        #cv2.imshow('findChessboardCorners', grayscale_image)

        c = cv2.waitKey(50) & 0xFF
        if c == 13 and found == True: # Enter
            # チェスボードコーナー検出情報を追加
            image_points.append(corner)
            object_points.append(pattern_points)
            capture_count += 1
        if c == 110: # N
            if messagebox.askyesno('askyesno','チェスボード撮影を終了し、カメラ内部パラメータを求めますか？'):
                cv2.destroyAllWindows()
                break
        if c == 27: # ESC
            if messagebox.askyesno('askyesno','プログラムを終了しますか？'):
                video_input.release()
                cv2.destroyAllWindows()
                exit()

    if len(image_points) > 0:
        # カメラ内部パラメータを計算
        print( 'calibrateCamera() start')
        rms, K, d, r, t = cv2.calibrateCamera(object_points,image_points,(frame.shape[1],frame.shape[0]),None,None)
        print( "RMS = ", rms)
        print( "K = \n", K)
        print( "d = ", d.ravel())
        np.savetxt("K.csv", K, delimiter =',',fmt="%0.14f") #カメラ行列の保存
        np.savetxt("d.csv", d, delimiter =',',fmt="%0.14f") #歪み係数の保存

        camera_mat = K
        dist_coef = d

        # 再投影誤差による評価
        mean_error = 0
        for i in xrange(len(object_points)):
            image_points2, _ = cv2.projectPoints(object_points[i], r[i], t[i], camera_mat, dist_coef)
            error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2) / len(image_points2)
            mean_error += error
        print( "total error: ", mean_error/len(object_points)) # 0に近い値が望ましい(魚眼レンズの評価には不適？)
    else:
        print( "findChessboardCorners() not be successful once")

# 歪み補正画像表示
if camera_mat != []:
    while(True):
        ret, frame = video_input.read()
        undistort_image = cv2.undistort(frame, camera_mat, dist_coef)

        cv2.imshow('original', frame)
        cv2.imshow('undistort', undistort_image)
        c = cv2.waitKey(50) & 0xFF
        if c==27: # ESC
            break

video_input.release()
cv2.destroyAllWindows()