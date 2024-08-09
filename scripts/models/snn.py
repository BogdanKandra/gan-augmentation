from math import prod

import torch.nn as nn
from torch import Tensor

from scripts import config


class SNN(nn.Module):
    def __init__(self, dataset: str) -> None:
        """ Class representing a shallow neural network, consisting of the
        Input and Output layers, and a single hidden layer
        
        Arguments:
            dataset (str): the name of the dataset to be used """
        super().__init__()

        # Compute the input and output feature sizes based on the specified dataset
        match dataset:
            case config.ClassifierDataset.FASHION_MNIST:
                self.in_features = prod(config.FASHION_MNIST_SHAPE)
                self.out_features = len(config.FASHION_MNIST_CLASS_LABELS)
            case config.ClassifierDataset.CIFAR_10:
                self.in_features = prod(config.CIFAR_10_SHAPE)
                self.out_features = len(config.CIFAR_10_CLASS_LABELS)
            case _:
                raise ValueError('Unimplemented dataset type')

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.out_features),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """ Tensor flow through the network for each dataset:
        Fashion-MNIST: (1,28,28) -> (1*28*28) -> (256) -> (10)
        CIFAR-10: (3,32,32) -> (3*32*32) -> (256) -> (10) """
        x = self.classifier(x)

        return x
