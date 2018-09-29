import ctypes
import numpy

runSimulation = ctypes.cdll.LoadLibrary ( './Simulator/Simulator.dll' ).simulate
createShot = ctypes.cdll.LoadLibrary ( './Simulator/Simulator.dll' ).createShot
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


import torch
def coordinates_to_plane(coordinates):
    #coors = [[int(round((x-0.14)/4.47 * 31)), int(round((y-2.906)/8.229 * 31))]
    #         for x, y in zip(coordinates[0][::2], coordinates[0][1::2])]
    number_of_coor = len(coordinates)
    coors = []
    for coordinate in coordinates:
        coors.append([])
        for x, y in zip(coordinate[0][::2], coordinate[0][1::2]):
            if x == 0 and y ==0:
                coors[-1].append([int(x), int(y)])
            else:
                coors[-1].append([int(round((x-0.14)/4.47 * 31)), int(round((y-2.906)/8.229 * 31))])

    plane = torch.zeros((number_of_coor, 2, 32, 32))
    for bat, coor in enumerate(coors):
        for i, (x, y) in enumerate(coor):
            if i % 2 == 0:
                plane[bat][0][y][x] = 1
            else:
                plane[bat][1][y][x] = 1

    print(plane.shape)
xy = numpy.zeros([32], numpy.float32)
a = simulate(xy, 14, 2.375, 4.88, 0, 0.145)
coordinates_to_plane([a,a, a])
# test = numpy.zeros((2,32,32))
# test[0][1][0] = 1
# print(test)