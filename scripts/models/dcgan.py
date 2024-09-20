import torch
import torch.nn as nn
from torch import Tensor

from scripts import config
from scripts.config import GeneratorDataset


class Generator(nn.Module):
    def __init__(self, dataset: GeneratorDataset) -> None:
        """ Class representing the generator part from the DCGAN. It consists of 4 blocks, each containing a
        Transposed Convolution layer, BatchNorm layer (except in the final layer), and ReLU activation. The output
        layer is a Transposed Convolution layer with tanh activation. Since this is a conditional GAN, the input
        consists of a noise vector, concatenated with the desired one-hot encoded label.

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
                self.z_dim = 64
                self.h_dim = 64
                self.in_channels = self.z_dim + len(config.FASHION_MNIST_CLASS_LABELS)
                self.out_channels = config.FASHION_MNIST_SHAPE[0]
                self.kernel_sizes = [3, 4, 3, 4]
                self.strides = [2, 1, 2, 2]
            case GeneratorDataset.CIFAR_10:
                self.z_dim = 128
                self.h_dim = 128
                self.in_channels = self.z_dim + len(config.CIFAR_10_CLASS_LABELS)
                self.out_channels = config.CIFAR_10_SHAPE[0]
                self.kernel_sizes = [3, 3, 3, 4]
                self.strides = [2, 2, 2, 2]
            case _:
                raise ValueError("Unsupported dataset type")

        self.generator = nn.Sequential(
            self._generator_block(self.in_channels, self.h_dim * 4,
                                  kernel_size=self.kernel_sizes[0], stride=self.strides[0]),
            self._generator_block(self.h_dim * 4, self.h_dim * 2,
                                  kernel_size=self.kernel_sizes[1], stride=self.strides[1]),
            self._generator_block(self.h_dim * 2, self.h_dim,
                                  kernel_size=self.kernel_sizes[2], stride=self.strides[2]),
            self._generator_block(self.h_dim, self.out_channels,
                                  kernel_size=self.kernel_sizes[3], stride=self.strides[3], final_layer=True)
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        """ Performs the forward pass of the generator network.
        The tensors flowing through the network have the following shapes:
        Fashion-MNIST: (64 + 10) -> (128) -> (256) -> (512) -> (1024) -> (1*28*28) -> (1, 28, 28)
        CIFAR-10: (128 + 10) -> (512) -> (1024) -> (2048) -> (4096) -> (3*32*32) -> (3, 32, 32)
        """
        z_and_labels = torch.cat((z, labels), dim=1)
        z_and_labels = z_and_labels.view(z_and_labels.shape[0], self.in_channels, 1, 1)

        return self.generator(z_and_labels)

    def _generator_block(self,
                         in_channels: int,
                         out_channels: int,
                         kernel_size: int = 3,
                         stride: int = 2,
                         final_layer: bool = False) -> None:
        """ Builds a block of the generator's model. """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.Tanh(),
            )


class Discriminator(nn.Module):
    def __init__(self, dataset: GeneratorDataset) -> None:
        """ Class representing the discriminator part from the vanilla GAN. It consists of 3 (or 4) blocks, each
        containing a Linear layer and a LeakyReLU activation. The output layer is a Linear layer. Since this is a
        conditional GAN, the input consists of an image, which is flattened and concatenated with the desired one-hot
        encoded label.

        Model hyperparameters:
            h_dim (int): the number of hidden features in the first discriminator block

        Arguments:
            dataset (GeneratorDataset): the name of the dataset to be used
        """
        super().__init__()

        # Set hyperparameters and compute the output features based on the specified dataset
        match dataset:
            case GeneratorDataset.FASHION_MNIST:
                self.in_shape = config.FASHION_MNIST_SHAPE
                self.h_dim = 512
                self.in_channels = self.in_shape[0] + len(config.FASHION_MNIST_CLASS_LABELS)
                self.kernel_sizes = [4, 4, 4]
                self.strides = [2, 2, 2]
            case GeneratorDataset.CIFAR_10:
                self.in_shape = config.CIFAR_10_SHAPE
                self.h_dim = 512
                self.in_channels = self.in_shape[0] + len(config.CIFAR_10_CLASS_LABELS)
                self.kernel_sizes = [5, 5, 4]
                self.strides = [2, 2, 2]
            case _:
                raise ValueError("Unsupported dataset type")

        self.discriminator = nn.Sequential(
            self._discriminator_block(self.in_channels, self.h_dim,
                                      kernel_size=self.kernel_sizes[0], stride=self.strides[0]),
            self._discriminator_block(self.h_dim, self.h_dim * 2,
                                      kernel_size=self.kernel_sizes[1], stride=self.strides[1]),
            self._discriminator_block(self.h_dim * 2, 1,
                                      kernel_size=self.kernel_sizes[2], stride=self.strides[2], final_layer=True)
        )

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """ Performs the forward pass of the discriminator network.
        The tensors flowing through the network have the following shapes:
        Fashion-MNIST: (1, 28, 28) -> (1*28*28) -> (784 + 10) -> (512) -> (256) -> (128) -> (1)
        CIFAR-10: (3, 32, 32) -> (3*32*32) -> (3072 + 10) -> (2048) -> (1024) -> (512) -> (256) -> (1)
        """
        x_and_labels = torch.cat((x, labels), dim=1)
        x = self.discriminator(x_and_labels)

        return x.view(len(x_and_labels), -1)

    def _discriminator_block(self,
                             in_channels: int,
                             out_channels: int,
                             kernel_size: int = 4,
                             stride: int = 2,
                             final_layer: bool = False) -> None:
        """ Builds a block of the discriminator's model. """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            )
