import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, use_dropout=False):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

        self.use_dropout = use_dropout

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))

        if self.droprate > 0:
            # Dropout always on training mode if use_dropout = True
            out = F.dropout(out, p=self.droprate, training=self.training or self.use_dropout)

        out = self.conv2(out)

        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, use_dropout=False):
        super(NetworkBlock, self).__init__()
        self.layer, self.layer_list = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,
                                                       use_dropout)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, use_dropout):
        layers = []

        for i in range(nb_layers):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, use_dropout))

        return nn.Sequential(*layers), layers

    def forward(self, x):
        out = self.layer_list[0](x)
        return out, self.layer(x)


'''
WideResNet base-model for CIFAR100
'''


class WideResNet_BaseModel(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, num_channel=3, dropRate=0.3, feature_extractor=False,
                 use_dropout=False):
        super(WideResNet_BaseModel, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(num_channel, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, use_dropout)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, use_dropout)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, use_dropout)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.clf = nn.Sequential(
            nn.Linear(nChannels[3], nChannels[3]),
            nn.ReLU(),
            nn.Linear(nChannels[3], num_classes)
        )

        self.num_input_channel = num_channel
        self.nChannels = nChannels[3]
        self.num_features = self.nChannels
        self.feature_extractor = feature_extractor
        self.use_dropout = use_dropout

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.feature_extractor:
            return self.features(x)
        else:
            out, fea_ist = self.features_before_clf(x)
            return self.clf(out), fea_ist

    def features(self, x):
        x = self.features_before_clf(x)
        for m in list(self.clf.children())[:-1]:
            x = m(x)
        return x

    def features_before_clf(self, x):
        x = self.conv1(x)
        fea1 = x.view(x.size(0), -1)
        out, x = self.block1(x)
        fea2_first = out.view(out.size(0), -1)
        fea2_second = x.view(x.size(0), -1)
        out, x = self.block2(x)
        fea3_first = out.view(out.size(0), -1)
        fea3_second = x.view(x.size(0), -1)
        out, x = self.block3(x)
        fea4_first = out.view(out.size(0), -1)
        fea4_second = x.view(x.size(0), -1)
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8 if self.num_input_channel == 3 else 7)
        fea5 = x.view(-1, self.nChannels)

        #Ignore shallow layer feature that is not helpful for meta-model
        return fea5, [fea3_first, fea3_second, fea4_first, fea4_second, fea5]


'''
WideResNet meta-model
'''


class WideResNet_MetaModel_combine(nn.Module):
    def __init__(self, fea_dim1, fea_dim2, fea_dim3, fea_dim4, fea_dim5):
        super(WideResNet_MetaModel_combine, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)

        self.classifier3_fc1 = nn.Linear(16384, 8192)
        self.classifier3_fc2 = nn.Linear(4096, 2048)
        self.classifier3_fc3 = nn.Linear(1024, 512)
        self.classifier3_fc4 = nn.Linear(256, 100)

        self.classifier3_fc1_second = nn.Linear(16384, 8192)
        self.classifier3_fc2_second = nn.Linear(4096, 2048)
        self.classifier3_fc3_second = nn.Linear(1024, 512)
        self.classifier3_fc4_second = nn.Linear(256, 100)

        self.classifier4_fc1 = nn.Linear(8192, 4096)
        self.classifier4_fc2 = nn.Linear(2048, 1024)
        self.classifier4_fc3 = nn.Linear(512, 256)
        self.classifier4_fc4 = nn.Linear(256, 100)

        self.classifier4_fc1_second = nn.Linear(8192, 4096)
        self.classifier4_fc2_second = nn.Linear(2048, 1024)
        self.classifier4_fc3_second = nn.Linear(512, 256)
        self.classifier4_fc4_second = nn.Linear(256, 100)

        self.classifier5_fc1 = nn.Linear(256, 100)

        self.classifier_final = nn.Linear(500, 100)

    def forward(self, fea3, fea3_second, fea4, fea4_second, fea5):
        fea3 = self.pooling(fea3)
        fea3 = F.relu(self.classifier3_fc1(fea3))
        fea3 = self.pooling(fea3)
        fea3 = F.relu(self.classifier3_fc2(fea3))
        fea3 = self.pooling(fea3)
        fea3 = F.relu(self.classifier3_fc3(fea3))
        fea3 = self.pooling(fea3)
        fea3 = F.relu(self.classifier3_fc4(fea3))

        fea3_second = self.pooling(fea3_second)
        fea3_second = F.relu(self.classifier3_fc1_second(fea3_second))
        fea3_second = self.pooling(fea3_second)
        fea3_second = F.relu(self.classifier3_fc2_second(fea3_second))
        fea3_second = self.pooling(fea3_second)
        fea3_second = F.relu(self.classifier3_fc3_second(fea3_second))
        fea3_second = self.pooling(fea3_second)
        fea3_second = F.relu(self.classifier3_fc4_second(fea3_second))

        fea4 = self.pooling(fea4)
        fea4 = F.relu(self.classifier4_fc1(fea4))
        fea4 = self.pooling(fea4)
        fea4 = F.relu(self.classifier4_fc2(fea4))
        fea4 = self.pooling(fea4)
        fea4 = F.relu(self.classifier4_fc3(fea4))
        fea4 = F.relu(self.classifier4_fc4(fea4))

        fea4_second = self.pooling(fea4_second)
        fea4_second = F.relu(self.classifier4_fc1_second(fea4_second))
        fea4_second = self.pooling(fea4_second)
        fea4_second = F.relu(self.classifier4_fc2_second(fea4_second))
        fea4_second = self.pooling(fea4_second)
        fea4_second = F.relu(self.classifier4_fc3_second(fea4_second))
        fea4_second = F.relu(self.classifier4_fc4_second(fea4_second))

        fea5 = F.relu(self.classifier5_fc1(fea5))

        fea = torch.cat((fea3, fea3_second, fea4, fea4_second, fea5), 1)
        z = self.classifier_final(fea)

        return z
