import itertools
import logging
import sys
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from scripts import config


def get_logger(name: str) -> logging.Logger:
    """ Creates and returns a Logger object for the requested script """
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)

    return logger


def is_perfect_square(number: int) -> bool:
    """ Checks whether the given number is a perfect square """
    if type(number) is not int:
        raise TypeError('Unexpected type for parameter "number" (Expected <int>, given <{}>)'.format(type(number)))

    return int(sqrt(number)) ** 2 == number


def get_maximum_batch_size(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int],
    dataset_size: int,
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    """ Searches for the maximum batch size that can be used for the given model on the given device.

    Arguments:
        model (nn.Module): the model to be trained
        device (torch.device): the device on which the model will be trained
        input_shape (Tuple[int, int, int]): the shape of the input data
        output_shape (Tuple[int]): the shape of the output data
        dataset_size (int): the size of the dataset
        max_batch_size (int, optional): the maximum batch size to be used
        num_iterations (int, optional): the number of training iterations to be performed
    """
    if device is torch.device('cpu'):
        raise ValueError('Batch size cannot be determined on CPU')

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()
    batch_size = 2

    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break

        if batch_size >= dataset_size:
            batch_size //= 2
            break

        try:
            for _ in range(num_iterations):
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                batch_loss = loss(targets, outputs)
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break

    del model, optimizer
    torch.cuda.empty_cache()

    return batch_size


def plot_results(subdirectory_name: str, history: Dict[str, List[float]]) -> None:
    """ Plots the training and validation accuracy and loss """
    training_accuracy = history['accuracy']
    validation_accuracy = history['val_accuracy']
    training_loss = history['loss']
    validation_loss = history['val_loss']

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Cross Entropy')
    plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')

    figure_path = config.CLASSIFIER_RESULTS_PATH / subdirectory_name / 'Training Results.png'
    plt.savefig(figure_path, dpi=300)
    plt.close()


def plot_confusion_matrix(conf_matrix: np.array, subdirectory_name: str, labels_list: List[str]) -> None:
    title = 'Confusion Matrix'

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels_list))
    plt.xticks(tick_marks, labels_list, rotation=45)
    plt.yticks(tick_marks, labels_list)

    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment='center',
                 color='white' if conf_matrix[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(config.CLASSIFIER_RESULTS_PATH / subdirectory_name / '{}.png'.format(title), dpi=300)
    plt.close()
