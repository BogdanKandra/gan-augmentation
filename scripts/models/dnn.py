from math import prod

import torch.nn as nn
from torch import Tensor

from scripts import config


class DNN(nn.Module):
    def __init__(self, dataset: str) -> None:
        """ Class representing a deep neural network, consisting of the
        Input and Output layers and 3 hidden layers in between, with a vanilla
        SGD as optimizer

        Arguments:
            dataset (str): the name of the dataset to be used """
        super().__init__()

        # Compute the input and output feature sizes based on the specified dataset
        match dataset:
            case config.DatasetType.FASHION_MNIST.name:
                self.in_features = prod(config.FASHION_MNIST_SHAPE)
                self.out_features = len(config.FASHION_MNIST_CLASS_LABELS)
            case config.DatasetType.CIFAR_10.name:
                self.in_features = prod(config.CIFAR_10_SHAPE)
                self.out_features = len(config.CIFAR_10_CLASS_LABELS)
            case _:
                raise ValueError('Unavailable dataset type')

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.out_features),
            nn.Softmax()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.classifier(x)  # (1*28*28) -> (256) -> (64) -> (16) -> (10) / (3*32*32) -> (256) -> (64) -> (16) -> (10)

        return x