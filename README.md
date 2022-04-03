#  mediapipe_to_vrchat

# Description
mediapipeで取得したデータをvrchatでフルトラするのに使えないか試した残骸。

# Requirement
- VMT - Virtual Motion Tracker
- mediapipe 0.8.9.1
- opencv-python 4.5.5.64

# Usage
## 事前準備
こちらのリポジトリを利用してカメラのキャリブレーションを行います。K_fisheye.csvとd_fisheye.csvというファイルが出力されるので、webcam_holistic_detection.pyと同じ階層に配置してください。

<https://github.com/Kazuhito00/OpenCV-CameraCalibration-Example>

## 姿勢推定の開始
コードを実行すると、カメラが起動し姿勢推定を開始します。また、同時に VTMへのOSCによる姿勢推定結果の送信が始まります。

```python webcam_holistic_detection.py```

## 姿勢推定結果の表示 
３次元空間で姿勢推定結果を確認することができます。
```python scatter_plot```
![](/images/line.png)

## 姿勢推定結果の表示2 
姿勢推定結果をリアルタイムにグラフで確認することができます。
```python line_plot.py```
![](/images/scatter.png)