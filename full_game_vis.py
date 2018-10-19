from gym_curling.env import Curling
import torch
import model
import numpy as np
import random
import utils
from gym_curling.Simulator import simulate as sim

import time


def main():
    model_file = ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = model.ResNet(model.ResidualBlock, [3, 4, 6, 3]).to(device)

    network.load_model(model_file)
    network.eval()

    data = []

    env = Curling(0.145)
    state = env.reset()

    for t in range(16):
        inputs = utils.to_input(data, t).unsqueeze(0)
        p_out, v_out = network(inputs.to(device))

        action = torch.argmax(p_out).item()
        action = utils.idx_to_action_xy(action)
        state, reward, done = env.step(action)

        topk_actions_idx2coor = []
        topk_actions = p_out.topk(32, 1, True, True)[1][0]
        for t_a in topk_actions:
            t_a = utils.idx_to_action_xy(t_a.item())
            topk_actions_idx2coor.append(t_a)

        env.render(True, topk_actions_idx2coor)
        # x = random.random() * 3.35 + 0.7
        # y = random.random() * 8.28
        #
        # curl = random.randint(0, 1)
        # state, reward, done = env.step([x, y, curl])
        # env.render(True, [[x, y, curl]])

        data = [[int(round(x / 4.75 * 31)), int(round(y / 11.28 * 31))] for x, y in zip(state[0][::2], state[0][1::2])]
        print(data)


if __name__ == '__main__':
    main()


