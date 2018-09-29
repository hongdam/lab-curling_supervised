from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from Simulator import Simulator as sim
import torch
import os
import numpy as np
import utils

import model


class CURL(Dataset):

    train_file = 'curl_train.txt'
    test_file = 'curl_test.txt'

    def __init__(self, path, train=True, transform=None, hammer=False):
        self.transform = transform
        self.train = train
        self.path = path
        self.is_hammer = hammer

        if self.train:
            file = os.path.join(path, self.train_file)
            self.train_data = np.loadtxt(file, delimiter=' ')
        else:
            file = os.path.join(path, self.test_file)
            self.test_data = np.loadtxt(file, delimiter=' ')

    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index, :32], self.train_data[index, 32:35]
        else:
            data, label = self.test_data[index, :32], self.test_data[index, 32:35]

        if self.is_hammer:
            order = 1
            turn = 15

            final_state = sim.simulate(data, turn, label[0], label[1], label[2], 0)[0]
            score = utils.get_score(final_state, 1)

        label = round(label[1] / 11.28 * 31) * 32 + \
                round(label[0] / 4.75 * 31) + \
                label[2] * 1024
        label = int(label)

        data = [[int(round(x/4.75 * 31)), int(round(y/11.28 * 31))]
                for x, y in zip(data[::2], data[1::2])]



        plane = np.zeros((2, 32, 32))
        ones_plane = np.ones((1, 32, 32))
        plane = np.concatenate((plane, ones_plane))
        zeros_plane = np.ones((1, 32, 32))

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

        if self.transform is not None:
            data = self.transform(data)

        plane = torch.FloatTensor(plane)

        return plane, label, score

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '\tNumber of datapoints = {}\n'.format(self.__len__())
        return fmt_str


import time
from torch import nn

learning_rate = 0.0000001

if __name__ == '__main__':
    # custom_dataset = CURL('./data', train=False, hammer=True)
    # for i in range(1000):
    #     print(custom_dataset[i][1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = model.ResNet(model.ResidualBlock, [2, 2, 2, 2]).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    # transformations = transforms.Compose([transforms.ToTensor()])
    custom_dataset = CURL('./data', train=True, hammer=True)
    loader = DataLoader(custom_dataset, batch_size=64, num_workers=4)


    for e in range(100):
        for i, data in enumerate(loader, 0):
            inputs, labels, scores = data

            p_out, v_out = network(inputs.to(device))

            one = criterion(v_out, torch.LongTensor(scores).to(device))
            two = criterion(p_out, torch.LongTensor(labels).to(device))
            loss = one + two

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print(torch.argmax(p_out[0]), labels[0])
                print(loss, one, two)
