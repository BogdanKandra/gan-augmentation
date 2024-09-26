import itertools
import logging
import sys
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import config
from scripts.config import NormalizationRange


def get_logger(name: str) -> logging.Logger:
    """ Creates and returns a Logger object for the requested script """
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)

    return logger


LOGGER = get_logger(__name__)


def is_perfect_square(number: int) -> bool:
    """ Checks whether the given number is a perfect square """
    if type(number) is not int:
        raise TypeError("Unexpected type for parameter 'number' (Expected <int>, given <{}>)".format(type(number)))

    return int(sqrt(number)) ** 2 == number


def get_maximum_classifier_batch_size(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int],
    dataset_size: int,
    max_batch_size: int = 2048,
    num_iterations: int = 5,
) -> int:
    """ Searches for the maximum batch size that can be used for the given classifier, on the given device.

    Arguments:
        model (nn.Module): the model to be trained
        device (torch.device): the device on which the model will be trained
        input_shape (Tuple[int, int, int]): the shape of the input data
        output_shape (Tuple[int]): the shape of the output data
        dataset_size (int): the size of the dataset
        max_batch_size (int, optional): the maximum batch size to be used
        num_iterations (int, optional): the number of training iterations to be performed
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()
    batch_size = 2

    while True:
        if batch_size >= dataset_size:
            LOGGER.info(f">>> Batch size exceeded dataset size: {dataset_size}!")
            batch_size //= 2
            break

        try:
            LOGGER.info(f"> Trying batch size {batch_size}...")

            for _ in range(num_iterations):
                # Train the classifier
                inputs = torch.randn((batch_size, *input_shape), dtype=torch.float32, device=device)
                targets = torch.randint(low=0, high=10, size=(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                batch_loss = loss(outputs, targets)
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if batch_size >= max_batch_size:
                LOGGER.info(f">>> Batch size reached maximum specified value: {max_batch_size}!")
                batch_size = max_batch_size
                break
            else:
                batch_size *= 2
        except RuntimeError:
            LOGGER.info(f">>> Batch size {batch_size} led to OOM error!")
            batch_size //= 2
            break

    LOGGER.info(">>> Emptying CUDA cache...")
    torch.cuda.empty_cache()

    LOGGER.info(f">>> Optimal batch size is set to {batch_size}.")
    return batch_size


def get_maximum_generator_batch_size(
    generator: nn.Module,
    discriminator: nn.Module,
    device: torch.device,
    gen_input_shape: int,
    disc_input_shape: Tuple[int, int, int],
    dataset_size: int,
    max_batch_size: int = 2048,
    num_iterations: int = 5,
) -> int:
    """ Searches for the maximum batch size that can be used for the given generator, on the given device.

    Arguments:
        generator (nn.Module): the generator model to be trained
        discriminator (nn.Module): the discriminator model to be trained
        device (torch.device): the device on which the model will be trained
        gen_input_shape (int): the shape of the generator input
        disc_input_shape (Tuple[int, int, int]): the shape of the discriminator input
        dataset_size (int): the size of the dataset
        max_batch_size (int, optional): the maximum batch size to be used
        num_iterations (int, optional): the number of training iterations to be performed
    """
    generator.to(device)
    generator.train()
    gen_optimizer = torch.optim.Adam(generator.parameters())

    discriminator.to(device)
    discriminator.train()
    disc_optimizer = torch.optim.Adam(discriminator.parameters())

    loss = nn.BCEWithLogitsLoss()
    batch_size = 2

    while True:
        if batch_size >= max_batch_size:
            LOGGER.info(f">>> Batch size reached maximum specified value: {max_batch_size}!")
            batch_size = max_batch_size
            break

        if batch_size >= dataset_size:
            LOGGER.info(f">>> Batch size exceeded dataset size: {dataset_size}!")
            batch_size //= 2
            break

        try:
            LOGGER.info(f"> Trying batch size {batch_size}...")

            for _ in range(num_iterations):
                # Train the discriminator
                noise = torch.randn((batch_size, gen_input_shape), device=device)
                labels = torch.randint(low=0, high=10, size=(batch_size,), device=device)
                labels_one_hot = F.one_hot(labels, num_classes=10)
                fake = generator(noise, labels_one_hot)
                predictions = discriminator(fake.detach(), labels_one_hot)
                loss_fakes = loss(predictions, torch.zeros_like(predictions))

                real = torch.randn((batch_size, *disc_input_shape), device=device)
                predictions = discriminator(real, labels_one_hot)
                loss_reals = loss(predictions, torch.ones_like(predictions))

                disc_loss = (loss_fakes + loss_reals) / 2
                disc_loss.backward()
                disc_optimizer.step()

                # Train the generator
                noise = torch.randn((batch_size, gen_input_shape), device=device)
                labels = torch.randint(low=0, high=10, size=(batch_size,), device=device)
                labels_one_hot = F.one_hot(labels, num_classes=10)
                fake = generator(noise, labels_one_hot)
                predictions = discriminator(fake, labels_one_hot)

                gen_loss = loss(predictions, torch.ones_like(predictions))
                gen_loss.backward()
                gen_optimizer.step()

                disc_optimizer.zero_grad()
                gen_optimizer.zero_grad()

            batch_size *= 2
        except RuntimeError:
            LOGGER.info(f">>> Batch size {batch_size} led to OOM error!")
            batch_size //= 2
            break

    LOGGER.info(f">>> Optimal batch size is set to {batch_size}.")
    torch.cuda.empty_cache()

    return batch_size


def unnormalize_image(image: torch.Tensor, normalization_range: NormalizationRange) -> torch.Tensor:
    """ Unnormalizes an image tensor.

    Arguments:
        image (torch.Tensor): the image tensor to be unnormalized
        normalization_range (NormalizationRange): the range of values obtained after normalization
    """
    if len(image.shape) == 3:
        # Image is given as (C, H, W)
        input_shape = (image.shape[0], 1, 1)
        channel_dim = 0
    else:
        # Image is given as (N, C, H, W)
        input_shape = (1, image.shape[1], 1, 1)
        channel_dim = 1

    match normalization_range:
        case NormalizationRange.TANGENT:
            # The image was normalized to [-1, 1]
            if image.shape[channel_dim] == 1:
                mean = torch.tensor([0.5]).reshape(input_shape).to(image.device)
                std = torch.tensor([0.5]).reshape(input_shape).to(image.device)
            else:
                mean = torch.tensor([0.5, 0.5, 0.5]).reshape(input_shape).to(image.device)
                std = torch.tensor([0.5, 0.5, 0.5]).reshape(input_shape).to(image.device)
        case NormalizationRange.IMAGENET:
            # The image was normalized with the ImageNet statistics
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(input_shape).to(image.device)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(input_shape).to(image.device)

    return image * std + mean


def initialize_weights(m: nn.Module) -> None:
    """ Initializes the weights of the model. """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def plot_classification_results(subdirectory_name: str, history: Dict[str, List[float]]) -> None:
    """ Plots the training and validation accuracy and loss for a classifier model """
    training_accuracy = history["accuracy"]
    validation_accuracy = history["val_accuracy"]
    training_loss = history["loss"]
    validation_loss = history["val_loss"]

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    plt.plot(training_accuracy, label="Training Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Categorical Cross Entropy")
    plt.ylim([0, max(plt.ylim())])
    plt.title("Training and Validation Loss")

    figure_path = config.CLASSIFIER_RESULTS_PATH / subdirectory_name / "Training Results.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()


def plot_generation_results(subdirectory_name: str, history: Dict[str, List[float]]) -> None:
    """ Plots the discriminator and generator losses """
    discriminator_loss = history["discriminator_loss"]
    generator_loss = history["generator_loss"]

    plt.figure(figsize=(18, 10))
    plt.plot(discriminator_loss, label="Discriminator Loss")
    plt.plot(generator_loss, label="Generator Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Categorical Cross Entropy")
    plt.ylim([0, max(plt.ylim())])
    plt.title("Discriminator and Generator Losses")

    figure_path = config.GENERATOR_RESULTS_PATH / subdirectory_name / "Training Results.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()


def plot_confusion_matrix(conf_matrix: np.array, subdirectory_name: str, labels_list: List[str]) -> None:
    title = "Confusion Matrix"

    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels_list))
    plt.xticks(tick_marks, labels_list, rotation=45)
    plt.yticks(tick_marks, labels_list)

    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(config.CLASSIFIER_RESULTS_PATH / subdirectory_name / "{}.png".format(title), dpi=300)
    plt.close()
