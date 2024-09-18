from scripts import utils
from scripts.generators.abstract_generator import AbstractGenerator

LOGGER = utils.get_logger(__name__)


class DDPM_Generator(AbstractGenerator):
    """ Class representing a generator for TorchVision datasets,
    using a Denoising Diffusion Probabilistic Model (DDPM) """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by <TBA>. """

    def build_model(self, compute_batch_size: bool = False) -> None:
        """ Defines the generator's model structure and stores it as an instance attribute.

        Arguments:
            compute_batch_size (bool, optional): whether to compute the maximum batch size for this model and device
        """

    def train_model(self, run_description: str) -> None:
        """ Defines the training parameters and runs the training loop for the model currently in memory. <TBA>
        is used as the optimizer, the loss function to be optimised is the <TBA>, and the
        measured metrics are <TBA>. An early stopping mechanism is used to prevent overfitting.

        Arguments:
            run_description (str): The description of the current run
        """

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by running it on the testing set. """
