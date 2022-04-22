from abc import ABC, abstractmethod
from math import sqrt
import scripts.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


LOGGER = utils.get_logger(__name__)


class FashionMNISTClassifier(ABC):
    """ Abstract class representing the blueprint all classifiers on the Fashion-MNIST dataset must follow """
    @abstractmethod
    def __init__(self):
        """ The base constructor loads the Fashion-MNIST dataset and stores it in instance attributes;
        it additionaly stores a model placeholder as instance attribute """
        (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data()
        self.model = None

    def display_dataset_information(self) -> None:
        """ Logs information about the dataset currently in memory """
        LOGGER.info('>>> Train Set Shape: X_train.shape={}, y_train.shape={}'.format(self.X_train.shape,
                                                                                     self.y_train.shape))
        LOGGER.info('>>> Test Set Shape: X_test.shape={}, y_test.shape={}'.format(self.X_test.shape,
                                                                                  self.y_test.shape))

    def display_dataset_sample(self, num_samples: int = 9) -> None:
        """ Displays some images from the dataset currently in memory """
        # Parameter validation
        max_samples = min(self.X_train.shape[0], 100)
        if num_samples > max_samples:
            raise ValueError('Maximum count of images to be displayed is {}'.format(max_samples))

        # Compute the plotting grid size as the next perfect square from num_samples
        if utils.is_perfect_square(num_samples):
            grid_size = int(sqrt(num_samples))
        else:
            next_perfect_square = num_samples + 1
            while not utils.is_perfect_square(next_perfect_square):
                next_perfect_square += 1
            grid_size = int(sqrt(next_perfect_square))

        # Plot the samples
        for i in range(num_samples):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(self.X_train[i], cmap=plt.get_cmap('gray'))

        plt.show()

    def display_model(self) -> None:
        """ Logs the summary of the model currently in memory """
        if self.model is not None:
            LOGGER.info(self.model.summary())
        else:
            LOGGER.info('>>> There is currently no model for this classifier')

    @abstractmethod
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory """
        pass

    # @abstractmethod
    # def build_model(self) -> None:
    #     """ Defines the classifier model structure and stores it as an instance attribute """
    #     pass
    #
    # @abstractmethod
    # def train_model(self) -> None:
    #     """ Performs the training of this classifier """
    #     pass
    #
    # @abstractmethod
    # def evaluate_model(self) -> None:
    #     """ Evaluates the model currently in memory """
    #     pass
    #
    # @abstractmethod
    # def export_model(self) -> None:
    #     """ Exports the model currently in memory in Tensorflow.js format """
    #     pass
    #
    # @abstractmethod
    # def run_model(self, image):
    #     """ Runs the model currently in memory on a sample image """
    #     pass
