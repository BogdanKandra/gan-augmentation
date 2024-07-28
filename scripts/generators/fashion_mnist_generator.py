from abc import ABC
from typing import Dict

from scripts import utils
from scripts.interfaces.fashion_mnist_model import FashionMNISTModel


LOGGER = utils.get_logger(__name__)


class FashionMNISTGenerator(FashionMNISTModel, ABC):
    """ Abstract class representing the blueprint all generators on the Fashion-MNIST dataset must follow """
    def __init__(self) -> None:
        """ TBA """
        pass

    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, 'preprocess_dataset') and callable(subclass.preprocess_dataset) and
                hasattr(subclass, 'build_model') and callable(subclass.build_model) and
                hasattr(subclass, 'train_model') and callable(subclass.train_model) or
                NotImplemented)

    def evaluate_model(self, hyperparams: Dict[str, int]) -> None:
        """ TBA """
        pass
