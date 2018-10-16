import ctypes
import numpy
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
dll_path = os.path.join(my_path, "./Simulator.dll")

runSimulation = ctypes.cdll.LoadLibrary (dll_path).simulate
createShot = ctypes.cdll.LoadLibrary (dll_path).createShot
createShot.argtypes = [ numpy.ctypeslib.ndpointer(numpy.float32, 1)]
runSimulation.argtypes = [numpy.ctypeslib.ndpointer(numpy.float32, 1)]


def simulate(xy, turn, x, y, curl, uncertainty):
    vector = numpy.zeros([3], numpy.float32)
    simulatedXY = numpy.zeros([37], numpy.float32)
    vector[0] = x
    vector[1] = y
    vector[2] = curl
    createShot(vector)
    simulatedXY[32] = turn
    simulatedXY[0:32] = xy[:]
    simulatedXY[36] = uncertainty
    simulatedXY[33:36] = vector[0:3]
    runSimulation(simulatedXY)
    simulatedXY[34] = curl

    return simulatedXY[0:32], simulatedXY[32:35]


import time
from multiprocessing import Process
import random


def mult_p_time_test(n):

    for _ in range(n):
        xy = numpy.zeros([32], numpy.float32)
        for i in range(16):
            # xy = simulate(xy, i, 2.375, 4.88, 0, 0.145)[0]
            xy = simulate(xy, i, random.random() * 4.75, random.random() * 11.28, random.randint(0, 1), 0.145)[0]


if __name__ == "__main__":

    s = time.time()
    for _ in range(100):
        xy = numpy.zeros([32], numpy.float32)
        for i in range(16):
            xy = simulate(xy, i, random.random()*4.75, random.random()*11.28, random.randint(0,1), 0.145)[0]
            # xy = simulate(xy, i, 2.375, 4.88, 0, 0.145)[0]
    print(time.time()-s)

    # for j in range(1,16):
    #     s = time.time()
    #     ns = [100] * j
    #     procs = []
    #     for i, n in enumerate(ns):
    #         proc = Process(target=mult_p_time_test, args=(n,))
    #         procs.append(proc)
    #         proc.start()
    #     for proc in procs:
    #         proc.join()
    #     print(time.time() - s)




# import torch
#
# from utils import coordinates_to_plane
#coordinates_to_plane([a,a, a])
# test = numpy.zeros((2,32,32))
# test[0][1][0] = 1
# print(test)