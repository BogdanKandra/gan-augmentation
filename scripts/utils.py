import itertools
import logging
from math import sqrt
import sys
from typing import List
from scripts import config
from matplotlib import pyplot as plt
import numpy as np


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


def plot_results(subdirectory_name: str, history: dict) -> None:
    """ Plots the training and validation accuracy and loss """
    training_accuracy = history['categorical_accuracy']
    validation_accuracy = history['val_categorical_accuracy']
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
