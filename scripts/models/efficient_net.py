import torch.nn as nn
from torch import Tensor
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from scripts import config


class EfficientNet(nn.Module):
    def __init__(self, dataset: str) -> None:
        """ Class representing a network based on the EfficientNetB0 pretrained
        network, using it as a fixed feature extractor, with a new classifier head
        consisting of Dropout and the Output layer

        Arguments:
            dataset (str): the name of the dataset to be used """
        super().__init__()

        # Compute the input and output feature sizes based on the specified dataset
        match dataset:
            case config.ClassifierDataset.FASHION_MNIST:
                self.out_features = len(config.FASHION_MNIST_CLASS_LABELS)
            case config.ClassifierDataset.CIFAR_10:
                self.out_features = len(config.CIFAR_10_CLASS_LABELS)
            case _:
                raise ValueError('Unimplemented dataset type')

        self.feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            # nn.BatchNorm2d(1280),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=self.out_features),
            nn.Softmax(dim=0)
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Tensor flow through the network for each dataset:
        Fashion-MNIST: (1,28,28) -> (3,224,224) -> (1280) -> (10)
        CIFAR-10: (3,32,32) -> (3,224,224) -> (1280) -> (10) """
        x = self.feature_extractor(x)

        return x
