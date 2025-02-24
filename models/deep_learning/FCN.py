import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, config, num_classes):
        super(FCN, self).__init__()
        input_channels, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding='valid')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding='valid')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(128, num_classes, kernel_size=3, padding='valid')
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x