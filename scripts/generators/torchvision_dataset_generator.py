import json
from abc import ABC
from math import sqrt
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision.utils import make_grid

from scripts import config, utils
from scripts.config import GeneratorDataset
from scripts.interfaces import TorchVisionDatasetModel


LOGGER = utils.get_logger(__name__)


class TorchVisionDatasetGenerator(TorchVisionDatasetModel, ABC):
    """ Abstract class representing the blueprint all generators on TorchVision datasets must follow """
    def __init__(self, dataset: GeneratorDataset) -> None:
        """ Loads the specified dataset and stores it in instance attributes. """
        # Determine the device to be used for storing the data, model, and metrics
        if torch.cuda.is_available():
            self.device: torch.device = torch.device('cuda')
            self.pin_memory: bool = True
            self.pin_memory_device: str = self.device.type
        else:
            self.device: torch.device = torch.device('cpu')
            self.pin_memory: bool = False
            self.pin_memory_device: str = ''

        # Load the specified dataset
        self.dataset_type: GeneratorDataset = dataset

        match self.dataset_type:
            case GeneratorDataset.FASHION_MNIST:
                train_dataset = FashionMNIST(root='data', train=True, download=True)
                self.dataset_shape = config.FASHION_MNIST_SHAPE
                self.class_labels = config.FASHION_MNIST_CLASS_LABELS
            case GeneratorDataset.CIFAR_10:
                train_dataset = CIFAR10(root='data', train=True, download=True)
                self.dataset_shape = config.CIFAR_10_SHAPE
                self.class_labels = config.CIFAR_10_CLASS_LABELS

        self.X, self.y = train_dataset.data, train_dataset.targets

        # Convert numpy arrays to torch tensors and move to GPU if available
        if type(self.X) is np.ndarray:
            self.X = torch.from_numpy(self.X).to(self.device)
            self.y = torch.tensor(self.y).to(self.device)

        self.preprocessed = False

        self.model = None
        self.discriminator = None
        self.hyperparams = {}
        self.gen_optimizer = None
        self.disc_optimizer = None
        self.loss = None

        self.results_subdirectory = None
        self.run_id = None
        self.training_history = {}
        self.evaluation_results = {}

    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, 'preprocess_dataset') and callable(subclass.preprocess_dataset) and
                hasattr(subclass, 'build_model') and callable(subclass.build_model) and
                hasattr(subclass, 'train_model') and callable(subclass.train_model) and
                hasattr(subclass, 'evaluate_model') and callable(subclass.evaluate_model))

    def display_dataset_information(self) -> None:
        """ Logs information about the dataset currently in memory. """
        LOGGER.info(f'>>> Training Data Information:\n\tshape: X.shape={self.X.shape}, y.shape={self.y.shape}\n\t'
                    f'dtype: X.dtype={self.X.dtype}, y.dtype={self.y.dtype}\n\t'
                    f'device: X.device={self.X.device}, y.device={self.y.device}')

    def display_dataset_sample(self, num_samples: int = 9) -> None:
        """ Displays random images from the dataset currently in memory. Maximum number of images to be displayed is
        min(100, dataset_size).

        Arguments:
            num_samples (int, optional): the number of images to be displayed
        """
        # Parameter validation
        max_samples = min(self.X.shape[0], 100)
        if num_samples > max_samples:
            raise ValueError(f'Maximum count of images to be displayed is {max_samples}')

        # Compute the plotting grid size as the next perfect square from num_samples
        if utils.is_perfect_square(num_samples):
            grid_size = int(sqrt(num_samples))
        else:
            next_perfect_square = num_samples + 1
            while not utils.is_perfect_square(next_perfect_square):
                next_perfect_square += 1
            grid_size = int(sqrt(next_perfect_square))

        # Compute the cmap used for displaying the images
        if self.X.shape[1] == 1:
            cmap = plt.get_cmap('gray')
        else:
            cmap = plt.get_cmap(None)

        # Plot random samples
        indices = [randrange(0, self.X.shape[0]) for _ in range(num_samples)]
        indices.extend([-1] * (grid_size ** 2 - num_samples))  # Pad with -1 for empty spaces

        _, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        for ax, i in zip(axes.flat, indices):
            if i == -1:
                ax.axis('off')
            else:
                # Image must be on the CPU and channels-last for matplotlib
                sample = self.X[i].to('cpu').permute(1, 2, 0)
                label = self.class_labels[self.y[i]]
                ax.imshow(sample, cmap=cmap)
                ax.set_title(label)
                ax.axis('off')

        plt.show()

    def display_model(self) -> None:
        """ Logs information about the GAN currently in memory. """
        if self.model is not None:
            LOGGER.info('>>> Generator components:')
            LOGGER.info(f'{self.model}\n\n')
            LOGGER.info('>>> Discriminator components:')
            LOGGER.info(f'{self.discriminator}\n\n')

            LOGGER.info('>>> Torchinfo Generator summary:')
            summary(
                self.model,
                input_size=(1, self.model.z_dim),
                col_names=["input_size", "output_size", "num_params",
                           "params_percent", "kernel_size", "mult_adds", "trainable"],
                device=self.device,
                verbose=1
            )
            print('\n\n')
            LOGGER.info('>>> Torchinfo Discriminator summary:')
            summary(
                self.discriminator,
                input_size=(1, *self.dataset_shape),
                col_names=["input_size", "output_size", "num_params",
                           "params_percent", "kernel_size", "mult_adds", "trainable"],
                device=self.device,
                verbose=1
            )
        else:
            LOGGER.info('>>> There is currently no model in memory for this GAN')

    def save_results(self) -> None:
        """ Saves the current training run results by plotting training and validation accuracy and loss,
        and generating the classification report and confusion matrix. """
        # Generate a file containing model information and parameters
        training_info_path = config.GENERATOR_RESULTS_PATH / self.results_subdirectory / 'Training Information.txt'
        with open(training_info_path, 'w') as f:
            f.write('GENERATOR MODEL ARCHITECTURE:\n')
            f.write('------------------------------\n')
            for line in str(self.model).split('\n'):
                f.write(f'{line}\n')

            f.write('\nDISCRIMINATOR MODEL ARCHITECTURE:\n')
            f.write('------------------------------\n')
            for line in str(self.discriminator).split('\n'):
                f.write(f'{line}\n')

            f.write('\nGENERATOR OPTIMIZER:\n')
            f.write('------------------------------\n')
            for line in str(self.gen_optimizer).split('\n'):
                f.write(f'{line}\n')

            f.write('\nDISCRIMINATOR OPTIMIZER:\n')
            f.write('------------------------------\n')
            for line in str(self.disc_optimizer).split('\n'):
                f.write(f'{line}\n')

            f.write('\nLOSS FUNCTION:\n')
            f.write('------------------------------\n')
            f.write(f'{str(self.loss)}\n')

            f.write('\nHYPERPARAMETERS:\n')
            f.write('------------------------------\n')
            f.write(f'Batch Size: {self.hyperparams["BATCH_SIZE"]}\n')
            f.write(f'Early Stopping Tolerance: {self.hyperparams["EARLY_STOPPING_TOLERANCE"]}\n')
            f.write(f'Learning Rate: {self.hyperparams["LEARNING_RATE"]}\n')
            f.write(f'Number of Epochs: {self.hyperparams["NUM_EPOCHS"]}\n')

        # Plot the discriminator and generator losses
        utils.plot_generation_results(self.results_subdirectory, self.training_history)

        # Save the training results
        results_path = config.GENERATOR_RESULTS_PATH / self.results_subdirectory / 'Training Results.txt'
        with open(results_path, 'w') as f:
            f.write(json.dumps(self.training_history, indent=4))

        # Save the evaluation results
        results_path = config.GENERATOR_RESULTS_PATH / self.results_subdirectory / 'Evaluation Results.txt'
        with open(results_path, 'w') as f:
            f.write(json.dumps(self.evaluation_results, indent=4))

    def export_model(self) -> None:
        """ Exports the model currently in memory in ONNX format. """
        generator_artifacts_path = config.GENERATORS_PATH / self.results_subdirectory
        generator_artifacts_path.mkdir()
        model_path = generator_artifacts_path / 'model.onnx'
        dummy_input = torch.randn(1, self.model.z_dim, device=self.device)
        self.model.eval()
        onnx_program = torch.onnx.dynamo_export(self.model, dummy_input)
        onnx_program.save(str(model_path))

    def _create_current_run_directory(self) -> None:
        """ Computes the run index of the current GAN training, creates a directory for the corresponding
        results and sets the name of the created directory as a class field. """
        current_run_gen_dataset = f"{self.__class__.__name__} {self.dataset_type.name}"
        training_runs = filter(lambda path: path.is_dir(), config.GENERATOR_RESULTS_PATH.iterdir())
        training_runs = list(map(lambda path: path.stem, training_runs))
        relevant_runs = list(filter(lambda name: name.startswith(current_run_gen_dataset), training_runs))

        if len(relevant_runs) == 0:
            current_run_dir_name = f"{current_run_gen_dataset} Run 1"
        else:
            run_numbers = [name.split(" ")[-1] for name in relevant_runs]
            latest_run = max(list(map(int, run_numbers)))
            current_run_dir_name = f"{current_run_gen_dataset} Run {latest_run + 1}"

        current_run_dir_path = config.GENERATOR_RESULTS_PATH / current_run_dir_name
        current_run_dir_path.mkdir()

        self.results_subdirectory = current_run_dir_name

    def _display_image_batch(self, images: torch.Tensor, num_samples: int = 25) -> None:
        """ Displays random images from the dataset currently in memory. Maximum number of images to be displayed is
        min(100, dataset_size).

        Arguments:
            num_samples (int, optional): the number of images to be displayed
        """
        if num_samples > images.shape[0]:
            num_samples = images.shape[0]
        images = images.detach().cpu()
        image_grid = make_grid(images[:num_samples], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()
