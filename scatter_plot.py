from tkinter import N
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtCore
import queue
import time
import multiprocessing
import asyncio
import websockets
import mmap
import pickle
import itertools


class LightPlotScatter:
    def __init__(
        self,
    ):
        app = pg.mkQApp("GLScatterPlotItem Example")
        self.w = gl.GLViewWidget()
        self.w.show()
        self.w.setWindowTitle("pyqtgraph example: GLScatterPlotItem")
        self.w.setCameraPosition(distance=20)

        g = gl.GLGridItem()
        self.w.addItem(g)

        g = gl.GLGridItem()
        g.rotate(90, 0, 1, 0)
        self.w.addItem(g)

        g = gl.GLGridItem()
        g.rotate(90, 1, 0, 0)
        self.w.addItem(g)

        self.phase = 0.0
        self.sp = None
        self.q = queue.Queue()
        self.txtitem = None

    def plot_init(self):
        # pos = np.random.random(size=(33, 3))
        pos = np.zeros((34, 3))

        self.sp = gl.GLScatterPlotItem(pos=pos, color=(1, 1, 1, 1), size=10)
        self.w.addItem(self.sp)

        xyz_list = [10, 0, -10]
        xyz_iter = itertools.product(xyz_list, repeat=len(xyz_list))

        for xyz in xyz_iter:
            self.txtitem = gl.GLTextItem(pos=xyz, text=f"{xyz}")
            self.w.addItem(self.txtitem)

        pos1 = (0, 0, 0)
        pos2 = (0, 0, 0)
        pos_init = np.array([pos1, pos2])

        kwargs = {"pos": pos_init, "color": (1, 1, 1, 1), "width": 1, "antialias": True}
        self.line_list = [[None for i in range(33)] for j in range(33)]
        self.point_list = [
            [12, 24],
            [24, 26],
            [26, 28],
            [28, 30],
            [30, 32],
            [32, 28],
            [12, 11],
            [23, 24],
            [11, 23],
            [23, 25],
            [25, 27],
            [27, 29],
            [29, 31],
            [31, 27],
            [12, 14],
            [14, 16],
            [16, 22],
            [16, 20],
            [20, 18],
            [18, 16],
            [11, 13],
            [13, 15],
            [15, 21],
            [15, 19],
            [19, 17],
            [17, 15],
            [10, 9],
            [8, 6],
            [6, 5],
            [5, 4],
            [4, 0],
            [0, 1],
            [1, 2],
            [3, 7],
        ]

        for point in self.point_list:
            self.line_list[point[0]][point[1]] = gl.GLLinePlotItem(**kwargs)
            self.w.addItem(self.line_list[point[0]][point[1]])

    def plot_get(self):
        pos = self.q.get()
        return pos

    def plot_put(self, pos):
        self.q.put(pos)

    def plot_update(self):
        with open("pos.pkl", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm.seek(0)
            pos = pickle.load(mm)

        self.sp.setData(pos=pos)

        pos_tuple = tuple(map(tuple, pos))

        for point in self.point_list:
            # print(pos_tuple[point[0]], pos_tuple[point[1]])
            pos_m = np.array([pos_tuple[point[0]], pos_tuple[point[1]]])
            # print(pos_m)
            kwargs = {
                "pos": pos_m,
                "color": (1, 1, 1, 1),
                "width": 1,
                "antialias": True,
            }
            self.line_list[point[0]][point[1]].setData(**kwargs)

    def plot_start(self):
        t = QtCore.QTimer()
        t.timeout.connect(self.plot_update)
        t.start(50)
        pg.exec()


if __name__ == "__main__":
    client = LightPlotScatter()
    client.plot_init()
    client.plot_start()
