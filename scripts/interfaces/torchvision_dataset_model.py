from abc import ABC, abstractmethod
from scripts import utils


LOGGER = utils.get_logger(__name__)


class TorchVisionDatasetModel(ABC):
    """ Abstract class representing the blueprint all classifiers and generators on TorchVision datasets must follow """
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, 'preprocess_dataset') and callable(subclass.preprocess_dataset) and
                hasattr(subclass, 'build_model') and callable(subclass.build_model) and
                hasattr(subclass, 'train_model') and callable(subclass.train_model) and
                hasattr(subclass, 'evaluate_model') and callable(subclass.evaluate_model) or
                NotImplemented)

    @abstractmethod
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory. """
        raise NotImplementedError

    @abstractmethod
    def build_model(self, compute_batch_size: bool = False) -> None:
        """ Defines the classifier's / generator's model structure and stores it as an instance attribute.

        Arguments:
            compute_batch_size (bool, optional): whether to compute the maximum batch size for this model and device
        """
        raise NotImplementedError

    @abstractmethod
    def train_model(self) -> None:
        """ Defines the training parameters and runs the training loop for the model currently in memory. """
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by running it on the testing set. """
        raise NotImplementedError

    # @abstractmethod
    # def run_model(self, image) -> None:
    #     """ Runs the model currently in memory on a sample image. """
    #     raise NotImplementedError
