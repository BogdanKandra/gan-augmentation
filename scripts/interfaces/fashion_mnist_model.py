from abc import ABC, abstractmethod
from scripts import utils
from typing import Dict


LOGGER = utils.get_logger(__name__)


class FashionMNISTModel(ABC):
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
        """ Defines the classifier model structure and stores it as an instance attribute """
        raise NotImplementedError

    @abstractmethod
    def train_model(self) -> None:
        """ Performs the training and evaluation of this classifier, on both the train set and the validation set """
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(self, hyperparams: Dict[str, int]) -> None:
        """ Evaluates the model currently in memory """
        raise NotImplementedError

    # @abstractmethod
    # def run_model(self, image) -> None:
    #     """ Runs the model currently in memory on a sample image """
    #     raise NotImplementedError
