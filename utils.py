import torch
import numpy as np


def get_score(state, turn):

    score = 0

    t_coor = np.array([2.375, 4.88])
    coors = [np.array([state[i], state[i+1]]) for i in range(0, 32, 2)]
    dists = [np.linalg.norm(t_coor-coor) for coor in coors]

    my = sorted(dists[turn::2])
    op = sorted(dists[1-turn::2])

    if my[0] < op[0]:
        for my_dist in my:
            if my_dist < op[0] and my_dist < 1.97:
                score += 1
            else:
                break
    else:
        for op_dist in op:
            if op_dist < my[0] and op_dist < 1.97:
                score -= 1
            else:
                break

    return score


def to_input(data, turn):
    plane = np.zeros((2, 32, 32))
    ones_plane = np.ones((1, 32, 32))
    plane = np.concatenate((plane, ones_plane))
    zeros_plane = np.zeros((1, 32, 32))

    order = turn % 2

    if order == 0:
        plane = np.concatenate((plane, ones_plane))
        plane = np.concatenate((plane, zeros_plane))
    else:
        plane = np.concatenate((plane, zeros_plane))
        plane = np.concatenate((plane, ones_plane))

    plane = np.concatenate((plane, np.zeros((8, 32, 32))))

    for i, c in enumerate(data):
        if c[0] == 0 and c[1] == 0:
            continue

        x, y = c

        if i % 2 == order:
            plane[0][y][x] = 1
        else:
            plane[1][y][x] = 1

    plane[5 + turn // 2] = np.ones((32, 32))

    plane = torch.FloatTensor(plane)

    return plane


def idx_to_action_xy(idx):
    curl = idx // 1024
    idx = idx - curl * 1024
    row = idx // 32
    col = idx % 32
    y = row/31. * 11.28
    x = col/31. * 4.75
    return [x, y, curl]