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
            if self.is_hammer:
                file = os.path.join(path, self.train_file)
                self.train_data = np.loadtxt(file, delimiter=' ')
            else:
                self.train_data = np.load('./train_data/train_data.npy')
        else:
            if self.is_hammer:
                file = os.path.join(path, self.test_file)
                self.test_data = np.loadtxt(file, delimiter=' ')
            else:
                self.test_data = np.load('./test_data/test_data.npy')

    def __getitem__(self, index):
        if self.train:
            if self.is_hammer:
                data, label = self.train_data[index, :32], self.train_data[index, 32:35]
            else:
                data = self.train_data[index, :32]
                label = self.train_data[index, 32:35]
                order = self.train_data[index, 35].astype(np.int)
                turn = self.train_data[index, 36].astype(np.int)

                score = self.train_data[index, 37] if order == 0 else -self.train_data[index, 37]
        else:
            if self.is_hammer:
                data, label = self.test_data[index, :32], self.test_data[index, 32:35]
            else:
                data = self.test_data[index, :32]
                label = self.test_data[index, 32:35]
                order = self.test_data[index, 35].astype(np.int)
                turn = self.test_data[index, 36].astype(np.int)
                score = self.test_data[index, 37] if order == 0 else -self.test_data[index, 37]



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
        zeros_plane = np.zeros((1, 32, 32))

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

        plane[5 + int(turn) // 2] = np.ones((32, 32))

        if order == 1:
            score = -score
        score += 8

        if self.transform is not None:
            data = self.transform(data)

        plane = torch.FloatTensor(plane)
        score = int(score)

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


from torch import nn
import visdom
learning_rate = 0.001
import tqdm


def test(net, device, test_loader, s_win, s_top1_win, l_win, l_top1_win, epoch):
    net.eval()
    label_correct = 0
    label_correct_top1 = 0
    score_correct = 0
    score_correct_top1 = 0

    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            inputs, labels, scores = data
            inputs = inputs.to(device)
            labels = torch.LongTensor(labels).to(device)
            scores = torch.LongTensor(scores).to(device)

            p_out, v_out = net(inputs)

            # top one
            pred = p_out.max(1, keepdim=True)[1]
            label_correct_top1 += pred.eq(labels.view_as(pred)).sum().item()
            #
            pred = v_out.max(1, keepdim=True)[1]
            score_correct_top1 += pred.eq(scores.view_as(pred)).sum().item()

            # top-5
            k = 5
            _, pred = p_out.topk(k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            label_correct += correct[:k].view(-1).float().sum(0, keepdim=True).item()

            # top-2
            _, pred = v_out.topk(2, 1, True, True)
            pred = pred.t()
            correct = pred.eq(scores.view(1, -1).expand_as(pred))
            score_correct += correct.view(-1).float().sum(0, keepdim=True).item()

    s_a = 100. * score_correct / len(test_loader.dataset)
    s_a_top1 = 100. * score_correct_top1 / len(test_loader.dataset)
    l_a = 100. * label_correct / len(test_loader.dataset)
    l_a_top1 = 100. * label_correct_top1 / len(test_loader.dataset)

    print('Score_acc: {:.3f}%, Label_acc: {:.2f}%'.format(s_a, l_a))
    print('Top1 Score_acc: {:.3f}%, Label_acc: {:.2f}%'.format(s_a_top1, l_a_top1))
    vis.line(np.asarray([s_a]), np.asarray([epoch]), win=s_win, update='append', opts=dict(title="score_top2"))
    vis.line(np.asarray([s_a_top1]), np.asarray([epoch]), win=s_top1_win, update='append', opts=dict(title="score_top1"))
    vis.line(np.asarray([l_a]), np.asarray([epoch]), win=l_win, update='append', opts=dict(title="shot_top5"))
    vis.line(np.asarray([l_a_top1]), np.asarray([epoch]), win=l_top1_win, update='append', opts=dict(title="shot_top1"))

    net.train()


if __name__ == '__main__':
    # custom_dataset = CURL('./data', train=False, hammer=True)
    # for i in range(1000):
    #     print(custom_dataset[i][1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = model.ResNet(model.ResidualBlock, [3, 4, 6, 3]).to(device)

    # network.load_model('.deep_fixed_9')

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    # [50, 120, 160]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 15, 30], gamma=0.1)
    # transformations = transforms.Compose([transforms.ToTensor()])

    train_dataset = CURL('./data', train=True, hammer=False)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)

    test_dataset = CURL('./data', train=False, hammer=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=4)

    vis = visdom.Visdom()
    s_win = vis.line(np.asarray([0]))
    s_top1_win = vis.line(np.asarray([0]))
    l_win = vis.line(np.asarray([0]))
    l_top1_win = vis.line(np.asarray([0]))

    vis_loss = vis.line(np.asarray([0]))
    vis_loss_v = vis.line(np.asarray([0]))
    vis_loss_p = vis.line(np.asarray([0]))

    current_iter = 0
    for e in range(500):
        print(e)
        scheduler.step()
        for i, data in enumerate(tqdm.tqdm(train_loader), 0):

            inputs, labels, scores = data

            p_out, v_out = network(inputs.to(device))

            one = criterion(v_out, scores.long().to(device))
            two = criterion(p_out, labels.to(device))
            loss = one + two

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if current_iter % 100 == 0:
                vis.line(loss.unsqueeze(0), np.asarray([current_iter]), win=vis_loss, update='append', opts=dict(title='loss'))
                vis.line(one.unsqueeze(0), np.asarray([current_iter]), win=vis_loss_v, update='append', opts=dict(title='loss_v'))
                vis.line(two.unsqueeze(0), np.asarray([current_iter]), win=vis_loss_p, update='append', opts=dict(title='loss_p'))
            current_iter += 1

        print(current_iter)
        print(torch.argmax(p_out[0]), labels[0])
        print(loss, one, two)

        test(network, device, test_loader, s_win, s_top1_win, l_win, l_top1_win, e+1)

        # if (e + 1) % 10 == 0:
        #     print("Save")
        #     network.save_model("./deep_"+str(e))

        network.save_model("./model_" + str(e))
