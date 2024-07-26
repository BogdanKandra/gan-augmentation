from math import prod

import torch.nn as nn
from torch import Tensor

from scripts import config


class CNN(nn.Module):
    def __init__(self, dataset: str) -> None:
        """ Class representing a convolutional neural network, consisting of 3
        convolutional blocks with pooling, dropout and L2 regularization,
        followed by 2 dense layers with dropout and Adam as optimizer

        Arguments:
            dataset (str): the name of the dataset to be used """
        super().__init__()

        # Compute the input and output feature sizes based on the specified dataset
        match dataset:
            case config.DatasetType.FASHION_MNIST.name:
                self.in_channels = config.FASHION_MNIST_SHAPE[0]
                self.out_features = len(config.FASHION_MNIST_CLASS_LABELS)
            case config.DatasetType.CIFAR_10.name:
                self.in_channels = config.CIFAR_10_SHAPE[0]
                self.out_features = len(config.CIFAR_10_CLASS_LABELS)
            case _:
                raise ValueError('Unavailable dataset type')

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.4)
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(in_features=prod([256, 4, 4]), out_features=256),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(in_features=256, out_features=self.out_features),
            # nn.Softmax()
        )

        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=self.in_features, out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=64),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=16),
        #     nn.ReLU(),
        #     nn.Linear(in_features=16, out_features=self.out_features),
        #     nn.Softmax()
        # )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.classifier(x)  # (1*28*28) -> (256) -> (64) -> (16) -> (10) / (3*32*32) -> (256) -> (64) -> (16) -> (10)

        return x
