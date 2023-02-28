import torch
import torch.nn as nn
import torch.nn.functional as F

'''
VGG16 base-model for CIFAR10
'''


class VGG16_BaseModel(nn.Module):
    def __init__(self):
        super(VGG16_BaseModel, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batchnorm4 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batchnorm5 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.batchnorm1(self.conv1_2(x)))
        x = self.pooling(x)
        fea1 = x.view(x.size(0), -1)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.batchnorm2(self.conv2_2(x)))
        x = self.pooling(x)
        fea2 = x.view(x.size(0), -1)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.batchnorm3(self.conv3_3(x)))
        x = self.pooling(x)
        fea3 = x.view(x.size(0), -1)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.batchnorm4(self.conv4_3(x)))
        x = self.pooling(x)
        fea4 = x.view(x.size(0), -1)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.batchnorm5(self.conv5_3(x)))
        x = self.pooling(x)
        fea5 = x.view(x.shape[0], -1)

        feature = F.relu(self.fc1(fea5))
        predict = self.fc2(feature)
        return predict, [fea1, fea2, fea3, fea4, fea5]


'''
VGG16 meta-model
'''


class VGG16_MetaModel_combine(nn.Module):
    def __init__(self, fea_dim1, fea_dim2, fea_dim3, fea_dim4, fea_dim5):
        super(VGG16_MetaModel_combine, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.classifier1_fc1 = nn.Linear(fea_dim1, 8192)
        self.classifier1_fc2 = nn.Linear(4096, 2048)
        self.classifier1_fc3 = nn.Linear(1024, 512)
        self.classifier1_fc4 = nn.Linear(256, 10)

        self.classifier2_fc1 = nn.Linear(fea_dim2, 4096)
        self.classifier2_fc2 = nn.Linear(2048, 1024)
        self.classifier2_fc3 = nn.Linear(512, 256)
        self.classifier2_fc4 = nn.Linear(256, 10)

        self.classifier3_fc1 = nn.Linear(fea_dim3, 2048)
        self.classifier3_fc2 = nn.Linear(1024, 512)
        self.classifier3_fc3 = nn.Linear(512, 256)
        self.classifier3_fc4 = nn.Linear(256, 10)

        self.classifier4_fc1 = nn.Linear(fea_dim4, 1024)
        self.classifier4_fc2 = nn.Linear(512, 256)
        self.classifier4_fc3 = nn.Linear(256, 10)

        self.classifier5_fc1 = nn.Linear(fea_dim5, 256)
        self.classifier5_fc2 = nn.Linear(256, 10)

        self.classifier_final = nn.Linear(5 * 10, 10)

    def forward(self, fea1, fea2, fea3, fea4, fea5):
        fea1 = F.relu(self.classifier1_fc1(fea1))
        fea1 = self.pooling(fea1)
        fea1 = F.relu(self.classifier1_fc2(fea1))
        fea1 = self.pooling(fea1)
        fea1 = F.relu(self.classifier1_fc3(fea1))
        fea1 = self.pooling(fea1)
        fea1 = F.relu(self.classifier1_fc4(fea1))

        fea2 = F.relu(self.classifier2_fc1(fea2))
        fea2 = self.pooling(fea2)
        fea2 = F.relu(self.classifier2_fc2(fea2))
        fea2 = self.pooling(fea2)
        fea2 = F.relu(self.classifier2_fc3(fea2))
        fea2 = F.relu(self.classifier2_fc4(fea2))

        fea3 = F.relu(self.classifier3_fc1(fea3))
        fea3 = self.pooling(fea3)
        fea3 = F.relu(self.classifier3_fc2(fea3))
        fea3 = F.relu(self.classifier3_fc3(fea3))
        fea3 = F.relu(self.classifier3_fc4(fea3))

        fea4 = F.relu(self.classifier4_fc1(fea4))
        fea4 = self.pooling(fea4)
        fea4 = F.relu(self.classifier4_fc2(fea4))
        fea4 = F.relu(self.classifier4_fc3(fea4))

        fea5 = F.relu(self.classifier5_fc1(fea5))
        fea5 = F.relu(self.classifier5_fc2(fea5))

        fea = torch.cat((fea1, fea2, fea3, fea4, fea5), 1)
        z = self.classifier_final(fea)
        return z
