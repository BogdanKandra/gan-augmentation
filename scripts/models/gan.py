from math import prod

import torch.nn as nn
from torch import Tensor

from scripts import config
from scripts.config import GeneratorDataset


class Generator(nn.Module):
    def __init__(self, dataset: GeneratorDataset) -> None:
        """ Class representing the generator part from the vanilla GAN. It consists of 4 blocks, each containing a
        Linear layer, BatchNorm layer, and ReLU activation. The output layer is a Linear layer with sigmoid activation.

        Model hyperparameters:
            z_dim (int): the number of features in the latent space / dimension of the noise vector
            h_dim (int): the number of hidden features in the first generator block

        Arguments:
            dataset (GeneratorDataset): the name of the dataset to be used
        """
        super().__init__()

        # Set hyperparameters and compute the output features based on the specified dataset
        match dataset:
            case GeneratorDataset.FASHION_MNIST:
                self.z_dim = 10
                self.h_dim = 128
                self.out_features = prod(config.FASHION_MNIST_SHAPE)
            case GeneratorDataset.CIFAR_10:
                self.z_dim = 40
                self.h_dim = 512
                self.out_features = prod(config.CIFAR_10_SHAPE)
            case _:
                raise ValueError('Unsupported dataset type')

        self.generator = nn.Sequential(
            self._generator_block(self.z_dim, self.h_dim),
            self._generator_block(self.h_dim, self.h_dim * 2),
            self._generator_block(self.h_dim * 2, self.h_dim * 4),
            self._generator_block(self.h_dim * 4, self.h_dim * 8),
            nn.Linear(self.h_dim * 8, self.out_features),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """ Performs the forward pass through the generator network. The tensors flow for each dataset as follows:
        Fashion-MNIST: (10) -> (128) -> (256) -> (512) -> (1024) -> (1*28*28)
        CIFAR-10: (40) -> (512) -> (1024) -> (2048) -> (4096) -> (3*32*32)
        """
        x = self.generator(z)

        return x

    def _generator_block(self, in_features: int, out_features: int) -> None:
        """ Builds a block of the generator's model. """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )


class Discriminator(nn.Module):
    def __init__(self, dataset: GeneratorDataset) -> None:
        """ Class representing the discriminator part from the vanilla GAN. It consists of 3 (or 4) blocks, each
        containing a Linear layer and a LeakyReLU activation. The output layer is a Linear layer.

        Model hyperparameters:
            h_dim (int): the number of hidden features in the first discriminator block

        Arguments:
            dataset (GeneratorDataset): the name of the dataset to be used
        """
        super().__init__()

        # Set hyperparameters and compute the output features based on the specified dataset
        match dataset:
            case GeneratorDataset.FASHION_MNIST:
                self.in_features = prod(config.FASHION_MNIST_SHAPE)
                self.h_dim = 512
                self.final_hidden_features = self.h_dim // 4
                self.discriminator_blocks = [
                    self._discriminator_block(in_features, out_features) for (in_features, out_features) in
                    [(self.in_features, self.h_dim),
                     (self.h_dim, self.h_dim // 2),
                     (self.h_dim // 2, self.h_dim // 4)]
                ]
            case GeneratorDataset.CIFAR_10:
                self.in_features = prod(config.CIFAR_10_SHAPE)
                self.h_dim = 2048
                self.final_hidden_features = self.h_dim // 8
                self.discriminator_blocks = [
                    self._discriminator_block(in_features, out_features) for (in_features, out_features) in
                    [(self.in_features, self.h_dim),
                     (self.h_dim, self.h_dim // 2),
                     (self.h_dim // 2, self.h_dim // 4),
                     (self.h_dim // 4, self.h_dim // 8)]
                ]
            case _:
                raise ValueError('Unsupported dataset type')

        self.discriminator = nn.Sequential(
            nn.Flatten(),
            *self.discriminator_blocks,
            nn.Linear(self.final_hidden_features, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Performs the forward pass through the discriminator network. The tensors flow for each dataset as follows:
        Fashion-MNIST: (1, 28, 28) -> (1*28*28) -> (512) -> (256) -> (128) -> (1)
        CIFAR-10: (3, 32, 32) -> (3*32*32) -> (2048) -> (1024) -> (512) -> (256) -> (1)
        """
        x = self.discriminator(x)

        return x

    def _discriminator_block(self, in_features: int, out_features: int) -> None:
        """ Builds a block of the discriminator's model. """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(negative_slope=0.2),
        )
