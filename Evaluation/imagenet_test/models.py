import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        # 加载预训练的ResNet18模型
        self.model = models.resnet18(pretrained=False)  # 不加载预训练权重

        # 修改最后一层全连接层的输出单元数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
