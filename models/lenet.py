import torch
import torch.nn as nn
import torch.nn.functional as F

'''
LeNet base-model for MNIST
'''


class LeNet_BaseModel(nn.Module):
    def __init__(self):
        super(LeNet_BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        fea1 = F.relu(self.conv1(x))
        fea1 = F.max_pool2d(fea1, 2)
        out = F.relu(self.conv2(fea1))
        out = F.max_pool2d(out, 2)
        fea2 = out.view(out.size(0), -1)
        out = F.relu(self.fc1(fea2))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        fea1 = fea1.view(fea1.size(0), -1)

        return out, [fea1, fea2]


'''
LeNet meta-model
'''


class LeNet_MetaModel_combine(nn.Module):
    def __init__(self, fea_dim1, fea_dim2):
        super(LeNet_MetaModel_combine, self).__init__()
        # Meta Model Layers
        self.classifier1_fc1 = nn.Linear(fea_dim1, 120)
        self.classifier1_fc2 = nn.Linear(120, 84)
        self.classifier1_fc3 = nn.Linear(84, 10)
        self.classifier2_fc1 = nn.Linear(fea_dim2, 120)
        self.classifier2_fc2 = nn.Linear(120, 84)
        self.classifier2_fc3 = nn.Linear(84, 10)

        self.classifier_final = nn.Linear(10 * 2, 10)

    def forward(self, x, y):
        # Meta Model
        x = F.relu(self.classifier1_fc1(x))
        x = F.relu(self.classifier1_fc2(x))
        x = F.relu(self.classifier1_fc3(x))
        y = F.relu(self.classifier2_fc1(y))
        y = F.relu(self.classifier2_fc2(y))
        y = F.relu(self.classifier2_fc3(y))
        z = torch.cat((x, y), 1)
        z = self.classifier_final(z)
        return z
