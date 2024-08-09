from abc import ABC

from scripts import utils
from scripts.config import GeneratorDataset
from scripts.interfaces import TorchVisionDatasetModel


LOGGER = utils.get_logger(__name__)


class TorchVisionDatasetGenerator(TorchVisionDatasetModel, ABC):
    """ Abstract class representing the blueprint all generators on TorchVision datasets must follow """
    def __init__(self, dataset: GeneratorDataset) -> None:
        """ TBA """
        pass

    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, 'preprocess_dataset') and callable(subclass.preprocess_dataset) and
                hasattr(subclass, 'build_model') and callable(subclass.build_model) and
                hasattr(subclass, 'train_model') and callable(subclass.train_model) and
                hasattr(subclass, 'evaluate_model') and callable(subclass.evaluate_model))
