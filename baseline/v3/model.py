import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import efficientnet
import math


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b4(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
        print(self.model)
        
    def forward(self, x):
        return self.model(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b0(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
        print(self.model)
        
    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
        print(self.model)

    def forward(self, x):
        return self.model(x)



# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. ?????? ?????? ???????????? parameter ??? num_claases ??? ??????????????????.
        2. ????????? ?????? ??????????????? ????????? ????????????.
        3. ????????? output_dimension ??? num_classes ??? ??????????????????.
        """

    def forward(self, x):
        """
        1. ????????? ????????? ?????? ??????????????? forward propagation ??? ??????????????????
        2. ????????? ?????? output ??? return ????????????
        """
        return x
