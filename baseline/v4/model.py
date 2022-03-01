import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg19_bn, efficientnet, resnet18


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


# Custom Model
class VGG19(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = vgg19_bn(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.model(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b0(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        # activation function : SiLU, sigmoid -> Xavier
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / (self.model.fc.weight.size(1)**0.5)
        self.model.fc.bias.data.uniform_(-stdv, stdv)
        # print(self.model)
        
    def forward(self, x):
        return self.model(x)

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b4(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        # activation function : SiLU, sigmoid -> Xavier
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / (self.model.fc.weight.size(1)**0.5)
        self.model.fc.bias.data.uniform_(-stdv, stdv)
        # print(self.model)
        
    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        ### INIT START ###
        # activation function : ReLU -> Kaiming
        torch.nn.init.kaiming_uniform_(self.model.fc.weight)
        ### INIT END ###
        stdv = 1. / (self.model.fc.weight.size(1)**0.5)
        self.model.fc.bias.data.uniform_(-stdv, stdv)
        # print(self.model)
        
    def forward(self, x):
        return self.model(x)

