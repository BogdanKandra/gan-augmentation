import torch.nn as nn
from torch import Tensor

from scripts import config
from scripts.config import ClassifierDataset


class CNN(nn.Module):
    def __init__(self, dataset: ClassifierDataset) -> None:
        """ Class representing a convolutional neural network, consisting of 3 convolutional blocks (with pooling,
        dropout and L2 regularization), followed by a decoder block (composed of 2 linear layers with dropout).

        Arguments:
            dataset (ClassifierDataset): the name of the dataset to be used
        """
        super().__init__()

        # Compute the input and output feature sizes based on the specified dataset
        match dataset:
            case ClassifierDataset.FASHION_MNIST:
                self.in_channels = config.FASHION_MNIST_SHAPE[0]
                self.decoder_features = 256 * (config.FASHION_MNIST_SHAPE[1] // 8) ** 2
                self.out_features = len(config.FASHION_MNIST_CLASS_LABELS)
            case ClassifierDataset.CIFAR_10:
                self.in_channels = config.CIFAR_10_SHAPE[0]
                self.decoder_features = 256 * (config.CIFAR_10_SHAPE[1] // 8) ** 2
                self.out_features = len(config.CIFAR_10_CLASS_LABELS)
            case _:
                raise ValueError('Unsupported dataset type')

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
            nn.Linear(in_features=self.decoder_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=self.out_features)
            # nn.Softmax(dim=0)  # Not needed here, since nn.CrossEntropyLoss() expects raw logits
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Performs the forward pass through the network. The tensors flow for each dataset as follows:
        Fashion-MNIST: (1,28,28) -> (64,14,14) -> (128,7,7) -> (256,3,3) ->
                       (256*3*3) -> (512) -> (10)
        CIFAR-10: (3,32,32) -> (64,16,16) -> (128,8,8) -> (256,4,4) ->
                  (256*4*4) -> (512) -> (10) """
        x = self.conv_block_1(x)  # (1,28,28) -> (64,14,14)  /  (3,32,32) -> (64,16,16)
        x = self.conv_block_2(x)  # (64,14,14) -> (128,7,7)  /  (64,16,16) -> (128,8,8)
        x = self.conv_block_3(x)  # (128,7,7) -> (256,3,3)   /  (128,8,8) -> (256,4,4)
        x = self.decoder(x)       # (256,3,3) -> (256*3*3) -> (512) -> (10)  /  (256,4,4) -> (256*4*4) -> (512) -> (10)

        return x
