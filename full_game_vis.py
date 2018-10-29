from gym_curling.env import Curling
import torch
import torch.nn.functional as F
import model
import numpy as np
import random
import utils
from gym_curling.Simulator import simulate as sim

import time


def expectation(x, prob):
    return torch.sum(torch.mul(x, prob), dim=(len(prob.shape)-1))


def to_prob(data):
    return F.softmax(data, dim=1)


def max_exp_shot(env, network, device, topk_coors, turn, x):
    number_of_sim = 16

    max_v = -99
    virtual_e = 0

    if turn != 15:

        # make input
        # virtual_inputs = torch.empty(0)
        list_for_cat = []
        for i, v_action in enumerate(topk_coors):
            for _ in range(number_of_sim):
                v_state, _ = env.virtual_step(v_action, 0.145)
                list_for_cat.append(utils.to_input(v_state, turn + 1).unsqueeze(0))
                # more time-consuming when cat in each iteration
                # virtual_inputs = torch.cat((virtual_inputs, utils.to_input(v_state, turn + 1).unsqueeze(0)), 0)
        virtual_inputs = torch.cat(list_for_cat, dim=0)

        # forward
        with torch.no_grad():
            _, virtual_v_out = network(virtual_inputs.to(device))
        virtual_v_out_prob = to_prob(virtual_v_out)
        # -x: because of op turn's value
        # 128x17 -> 128 (#sim x #topk)
        virtual_e = expectation(-x, virtual_v_out_prob)
        virtual_e = virtual_e.split(number_of_sim)
        virtual_e = [x.mean() for x in virtual_e]

        # hum..
        highlight = virtual_e.index(max(virtual_e))
        max_action = topk_coors[highlight]

    else:
        for i, v_action in enumerate(topk_coors):
            for _ in range(number_of_sim):
                v_state, _ = env.virtual_step(v_action, 0.145)
                # order is 1 because of last shot
                virtual_e += utils.get_score(v_state, 1)

            virtual_e = virtual_e / number_of_sim
            print(v_action, virtual_e)
            if max_v < virtual_e:
                max_v = virtual_e
                max_action = v_action
                highlight = i

    return max_action, highlight


def top_1_shot(policy):
    action = torch.argmax(policy).item()
    action = utils.idx_to_action_xy(action)

    return action, 0


def random_shot():
    x = random.random() * 3.35 + 0.7
    y = random.random() * 8.28

    curl = random.randint(0, 1)

    return x, y, curl

def main():
    number_of_games = 1

    model_file = './model_1'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = model.ResNet(model.ResidualBlock, [3, 4, 6, 3]).to(device)

    network.load_model(model_file)
    network.eval()

    x = torch.FloatTensor(range(-8, 8 + 1)).to(device)

    env = Curling(0.145)

    rewards = []
    for _ in range(number_of_games):
        state = env.reset()

        for t in range(16):
            inputs = utils.to_input(state[0], t).unsqueeze(0)
            p_out, v_out = network(inputs.to(device))

            topk_actions_idx2coor = []
            topk_actions = p_out.topk(32, 1, True, True)[1][0]
            for t_a in topk_actions:
                t_a = utils.idx_to_action_xy(t_a.item())
                topk_actions_idx2coor.append(t_a)

            if t % 2 == 1:
                action, highlight = max_exp_shot(env, network, device, topk_actions_idx2coor, t, x)
            else:
                action, highlight = top_1_shot(p_out)
            state, reward, done = env.step(action)
            env.render(True, topk_actions_idx2coor, highlight)

        print(-reward)
        rewards.append(-reward)
    print(rewards)

    
if __name__ == '__main__':
    main()


