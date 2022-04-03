import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import mmap
import pickle


class DataStore:
    def __init__(self):
        self.data = []

    def add_data(self, new_data):
        self.data.append(new_data)
        self.data = self.data[-100:]
        last_data = self.data
        data_idx = np.arange(len(last_data))
        return last_data, data_idx


# from pykalman import KalmanFilter


# def filtered_kalman(values):
#     kf = KalmanFilter(
#         transition_matrices=np.array([[1, 1], [0, 1]]),
#         transition_covariance=0.0001 * np.eye(2),
#     )  # np.eyeは単位行列
#     smoothed = kf.em(values).smooth(values)[0]
#     filtered = kf.em(values).filter(values)[0]
#     return smoothed, filtered


class LightPlotLine:
    def __init__(self):
        self.app = pg.mkQApp()
        self.mw = QtWidgets.QMainWindow()
        self.mw.setWindowTitle("pyqtgraph example: PlotWidget")
        self.mw.resize(800, 800)
        self.cw = QtWidgets.QWidget()
        self.mw.setCentralWidget(self.cw)
        self.l = QtWidgets.QVBoxLayout()
        self.cw.setLayout(self.l)

        self.data = []
        self.obj_list = []
        self.obj_list = [[None for i in range(33)] for j in range(33)]
        # self.data_store_list = []
        self.data_store_list = [[None for i in range(33)] for j in range(33)]

    def plot_init(self):
        # for i in range(3):
        for i in [0]:
            for j in range(3):
                self.pw1 = pg.PlotWidget()
                if j == 0:
                    self.pw1.setYRange(-1.5, 1.5)
                elif j == 1:
                    self.pw1.setYRange(2, 0)
                elif j == 2:
                    self.pw1.setYRange(0, 2)
                else:
                    pass
                self.l.addWidget(self.pw1)
                self.p1 = self.pw1.plot()
                # self.p1.setPen((100, 100, 100))

                self.obj_list[i][j] = self.p1
                data_store = DataStore()
                # self.data_store_list.append(data_store)
                self.data_store_list[i][j] = data_store

        # self.pw1 = pg.PlotWidget()
        # self.l.addWidget(self.pw1)
        # self.pw2 = pg.PlotWidget()
        # self.l.addWidget(self.pw2)
        # self.pw3 = pg.PlotWidget()
        # self.l.addWidget(self.pw3)

        # self.p1 = self.pw1.plot()
        # # self.p1.setPen((100, 100, 100))
        # self.p2 = self.pw2.plot()
        # # self.p2.setPen((100, 100, 100))
        # self.p3 = self.pw3.plot()
        # # self.p3.setPen((100, 100, 100))

        # self.pw1.setLabel("left", "Value", units="V")
        # self.pw1.setLabel("bottom", "Time", units="s")
        # self.pw2.setLabel("left", "Value", units="V")
        # self.pw2.setLabel("bottom", "Time", units="s")
        # self.pw3.setLabel("left", "Value", units="V")
        # self.pw3.setLabel("bottom", "Time", units="s")

        self.mw.show()

    def rand(self):
        random_float = np.random.rand()
        self.data.append(random_float)
        last_data = self.data[-100:]
        data_len = np.arange(len(last_data))
        return last_data, data_len

    def updateData(self):
        yd, xd = self.rand()

        with open("pos.pkl", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm.seek(0)
            pos = pickle.load(mm)

        # pos_tuple = tuple(map(tuple, pos))
        # for point in self.point_list:
        #     pos_m = np.array([pos_tuple[point[0]], pos_tuple[point[1]]])

        # with open("pos.pkl", "r+b") as f:
        #     mm = mmap.mmap(f.fileno(), 0)
        #     mm.seek(0)
        #     pos = pickle.load(mm)

        # self.p1.setData(y=yd, x=xd)
        # self.p2.setData(y=yd, x=xd)
        # self.p3.setData(y=yd, x=xd)

        # for i, obj in enumerate(self.obj_list):
        #     obj.setData(y=yd * i, x=xd*i)

        # for i in range(3):
        for i in [0]:
            for j in range(3):
                data_store = self.data_store_list[i][j]
                yd, xd = data_store.add_data(pos[i][j])
                if j == 1:
                    self.obj_list[i][2].setData(y=yd, x=xd)
                elif j == 2:
                    self.obj_list[i][1].setData(y=yd, x=xd)
                else:
                    self.obj_list[i][0].setData(y=yd, x=xd)

    def plot_start(self):
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.updateData)
        self.t.start(50)


if __name__ == "__main__":
    client = LightPlotLine()
    client.plot_init()
    client.plot_start()
    pg.exec()
