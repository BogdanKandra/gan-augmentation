from abc import ABC, abstractmethod
from scripts import utils
from typing import Dict, List


LOGGER = utils.get_logger(__name__)


class TorchVisionDatasetModel(ABC):
    """ Abstract class representing the blueprint all classifiers and generators
    on the Fashion-MNIST dataset must follow """
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, 'preprocess_dataset') and callable(subclass.preprocess_dataset) and
                hasattr(subclass, 'build_model') and callable(subclass.build_model) and
                hasattr(subclass, 'train_model') and callable(subclass.train_model) and
                hasattr(subclass, 'evaluate_model') and callable(subclass.evaluate_model) or
                NotImplemented)

    @abstractmethod
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory """
        raise NotImplementedError

    @abstractmethod
    def build_model(self) -> None:
        """ Defines the classifier / generator model structure and stores it as an instance attribute """
        raise NotImplementedError

    @abstractmethod
    def train_model(self) -> Dict[str, List[float]]:
        """ Defines the training parameters and runs the training loop for the model currently in memory

        Returns:
            Dict[str, List[float]]: dictionary containing the loss values and the accuracy,
                precision, recall and F1 score results, both on the training and validation sets
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(self) -> Dict[str, float]:
        """ Evaluates the model currently in memory by running it on the testing set

        Returns:
            Dict[str, float]: dictionary containing the loss value and the accuracy,
                precision, recall and F1 score results on the testing set
        """
        raise NotImplementedError

    # @abstractmethod
    # def run_model(self, image) -> None:
    #     """ Runs the model currently in memory on a sample image """
    #     raise NotImplementedError
