import cv2 
import sys

aruco = cv2.aruco #arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def arGenerator():
    for i in range(50):
        generator = aruco.drawMarker(dictionary, i, 100)
        cv2.imwrite(str(i) + '.png', generator)

def arReader(): 
    cap = cv2.VideoCapture(0) #ビデオキャプチャの開始

    while True:
        ret, frame = cap.read() #ビデオキャプチャから画像を取得
        # print(frame.shape)
        # print(type(frame))

        Height, Width = frame.shape[:2] #sizeを取得

        #sizeを半分に縮小
        halfHeight = Height / 2.0
        halfWidth = Width /2.0

        # print(type(halfHeight),type(halfWidth))
        # print(halfHeight,halfWidth)
        # imghalf = cv2.resize(frame,(halfWidth),(halfHeight))
        imghalf = frame

        corners, ids, rejectedImgPoints = aruco.detectMarkers(imghalf, dictionary) #マーカを検出
        aruco.drawDetectedMarkers(imghalf, corners, ids, (0,255,0)) #検出したマーカに描画する
        cv2.imshow('drawDetectedMarkers', imghalf) #マーカが描画された画像を表示
        cv2.waitKey(1) #キーボード入力の受付

    cap.release() #ビデオキャプチャのメモリ解放
    cv2.destroyAllWindows() #すべてのウィンドウを閉じる

if __name__ == '__main__':
    args = sys.argv
    ar = args[1]
    if ar == "Generator":
        arGenerator()
    elif ar == "Reader":
        arReader()
    else:
        print("Please enter valid argument")