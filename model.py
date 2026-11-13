import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model(num_classes: int = 50, freeze_backbone: bool = False) -> nn.Module:
    # загружаем предобученную ResNet-18 с актуальными весами ImageNet
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # опционально замораживаем веса сверточной части (backbone)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # узнаем количество входных признаков последнего слоя (у ResNet-18 это 512)
    in_features = model.fc.in_features

    # заменяем классификатор ("голову") на наш слой с Dropout против переобучения
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == '__main__':
    # быстрая проверка архитектуры
    net = get_model(num_classes=50)
    print(net.fc)