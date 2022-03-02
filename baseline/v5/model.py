import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import efficientnet
import math
from timm.models import vision_transformer


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

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b4(pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1792, out_features=512, bias=True)
        self.activation = MemoryEfficientSwish()
        self.classifier = nn.Linear(in_features = 512, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=512, bias=True)
        self.activation = MemoryEfficientSwish()
        self.classifier = nn.Linear(in_features = 512, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x

class EfficientNetB1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b1(pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=512, bias=True)
        self.activation = MemoryEfficientSwish()
        self.classifier = nn.Linear(in_features = 512, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x

class EfficientNetB2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b2(pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1536, out_features=512, bias=True)
        self.activation = MemoryEfficientSwish()
        self.classifier = nn.Linear(in_features = 512, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x

class EfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b3(pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1536, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        torch.nn.init.kaiming_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


class RegressionEfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = efficientnet.efficientnet_b4(pretrained=True)
        self.model.classifier[1] = nn.AvgPool2d(3, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm = nn.BatchNorm2d(num_features=1792)
        self.fc = nn.Linear(1792, 1)
    
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = vision_transformer.vit_base_patch32_384(pretrained=True)
        self.model.head = nn.Linear(768, num_classes, True)
        print(self.model)

    def forward(self, x):
        return self.model(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x
