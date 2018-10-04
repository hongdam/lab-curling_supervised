import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 128
        self.conv = conv3x3(13, 128) # my stone, enemy's stone, 1, order0, order1, turn 0~7
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 128, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1])
        self.layer3 = self.make_layer(block, 128, layers[2])
        self.layer4 = self.make_layer(block, 128, layers[3])
        self.value_conv = nn.Conv2d(128, 1, 3, 1, 1)
        self.value_fc = nn.Linear(32 * 32, 17)
        # self.value_softmax = nn.Softmax(dim=1)

        self.policy_conv1 = nn.Conv2d(128, 2, 3, 1, 1)
        self.policy_conv2 = nn.Conv2d(2, 2, 3, 1, 1)
        # self.policy_softmax = nn.Softmax(dim=1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        p_out = self.policy_conv1(out)
        v_out = self. value_conv(out)

        v_out = v_out.view(out.size(0), -1)
        v_out = self.value_fc(v_out)
        # v_out = self.value_softmax(v_out)

        p_out = self.policy_conv2(p_out)
        p_out = p_out.view(out.size(0), -1)
        # p_out = self.policy_softmax(p_out)
        return p_out, v_out  # 1 x 2048, 1 x 17


    def save_model(self, f_name):
        torch.save(self.state_dict(), f_name)


    def load_model(self, f_name):
        self.load_state_dict(torch.load(f_name))