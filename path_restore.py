import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = None

        self.act = activation

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.down is not None:
            residual = self.down(residual)
        x = self.act(x + residual)

        return x


class PathFinder(nn.Module):
    def __init__(self, in_channels, path_action_num, pathfinder_conv_num=4, hidden_dim=64,
                 kernel_size=3, stride=1, padding=1, activation=nn.ReLU()):
        super(PathFinder, self).__init__()
        layers = []
        channels = [in_channels] + [path_action_num] * pathfinder_conv_num
        for i in range(pathfinder_conv_num):
            layers.append(
                nn.Conv2d(channels[i], channels[i+1], kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(activation)
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(path_action_num, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, path_action_num)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.gap(x)
        x = x.view(-1, x.shape[1])
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class DynamicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ReLU(),
                 path_finder=None, path_action_num=2):
        super(DynamicBlock, self).__init__()
        assert(path_finder is not None)
        self.path_finder = path_finder
        self.shared_block = ResidualBlock(
            in_channels, out_channels, kernel_size, stride, padding, activation)
        self.path = nn.ModuleList()
        out_channels_inter = out_channels
        for i in range(path_action_num):
            if i == 0:
                self.path.append(nn.Identity())
            else:
                self.path.append(ResidualBlock(
                    out_channels_inter, out_channels, kernel_size, stride, padding, activation))

    def forward(self, x):
        shared = self.shared_block(x)
        path_logits = self.path_finder(x)
        if self.train:
            path_logits = torch.softmax(path_logits, dim=1)
            action = torch.multinomial(path_logits, 1)
        else:
            action = torch.argmax(path_logits, dim=1)
        ret = []
        for i in range(path_logits.shape[0]):
            ret.append(self.path[action[i]](shared[i:i+1]))
        return torch.cat(ret, dim=0)


class PathRestore(nn.Module):
    def __init__(self, in_channels, out_channels, block_channels=64, activation=nn.ReLU(),
                 path_action_num=2, dynamic_block_num=6, pathfinder_conv_num=2):
        super(PathRestore, self).__init__()
        self.act = activation
        self.conv1 = nn.Conv2d(in_channels, block_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(block_channels)
        self.path_finder = PathFinder(block_channels, path_action_num, pathfinder_conv_num, activation=self.act)
        layers = []
        for i in range(dynamic_block_num):
            layers.append(DynamicBlock(
                block_channels, block_channels, 3, 1, 1, activation, self.path_finder, path_action_num))
            layers.append(nn.BatchNorm2d(block_channels))
            layers.append(activation)
        self.dynamicblocks = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(block_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dynamicblocks(x)
        x = self.conv2(x)
        return x


def test():
    model = PathRestore(3, 3)
    inputs = torch.randn(2, 3, 63, 63)
    output = model(inputs)
    print(model)
    print(output.shape)


if __name__ == "__main__":
    test()
