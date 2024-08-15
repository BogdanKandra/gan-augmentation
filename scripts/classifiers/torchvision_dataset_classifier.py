import json
from abc import ABC
from math import sqrt
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk_metrics
import torch
from torchinfo import summary
from torchvision.datasets import CIFAR10, FashionMNIST

from scripts import config, utils
from scripts.config import ClassifierDataset
from scripts.interfaces import TorchVisionDatasetModel


LOGGER = utils.get_logger(__name__)


class TorchVisionDatasetClassifier(TorchVisionDatasetModel, ABC):
    """ Abstract class representing the blueprint all classifiers on TorchVision datasets must follow """
    def __init__(self, dataset: ClassifierDataset) -> None:
        """ Loads the specified dataset and stores it in instance attributes. """
        self.dataset_type = dataset

        match self.dataset_type:
            case ClassifierDataset.FASHION_MNIST:
                train_dataset = FashionMNIST(root='data', train=True, download=True)
                test_dataset = FashionMNIST(root='data', train=False, download=True)
                self.dataset_shape = config.FASHION_MNIST_SHAPE
                self.class_labels = config.FASHION_MNIST_CLASS_LABELS
            case ClassifierDataset.CIFAR_10:
                train_dataset = CIFAR10(root='data', train=True, download=True)
                test_dataset = CIFAR10(root='data', train=False, download=True)
                self.dataset_shape = config.CIFAR_10_SHAPE
                self.class_labels = config.CIFAR_10_CLASS_LABELS

        self.X_train, self.y_train = train_dataset.data, train_dataset.targets
        self.X_test, self.y_test = test_dataset.data, test_dataset.targets

        if type(self.X_train) is np.ndarray:
            self.X_train = torch.from_numpy(self.X_train)
            self.X_test = torch.from_numpy(self.X_test)
            self.y_train = torch.tensor(self.y_train)
            self.y_test = torch.tensor(self.y_test)

        validation_size = int(config.VALID_SET_PERCENTAGE * len(self.X_train))
        self.X_valid = self.X_train[: validation_size]
        self.y_valid = self.y_train[: validation_size]
        self.X_train = self.X_train[validation_size:]
        self.y_train = self.y_train[validation_size:]

        self.preprocessed = False

        self.model = None
        self.hyperparams = {}
        self.optimizer = None
        self.loss = None

        self.training_history = {}
        self.evaluation_results = {}
        self.results_subdirectory = None

    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, 'preprocess_dataset') and callable(subclass.preprocess_dataset) and
                hasattr(subclass, 'build_model') and callable(subclass.build_model) and
                hasattr(subclass, 'train_model') and callable(subclass.train_model) and
                hasattr(subclass, 'evaluate_model') and callable(subclass.evaluate_model))

    def display_dataset_information(self) -> None:
        """ Logs information about the dataset currently in memory. """
        LOGGER.info(f'>>> Train Set Shape: X_train.shape={self.X_train.shape}, y_train.shape={self.y_train.shape}')
        LOGGER.info(f'>>> Train Set dtype: X_train.dtype={self.X_train.dtype}, y_train.dtype={self.y_train.dtype}')
        LOGGER.info(f'>>> Validation Set Shape: X_valid.shape={self.X_valid.shape}, y_valid.shape={self.y_valid.shape}')
        LOGGER.info(f'>>> Validation Set dtype: X_valid.dtype={self.X_valid.dtype}, y_valid.dtype={self.y_valid.dtype}')
        LOGGER.info(f'>>> Test Set Shape: X_test.shape={self.X_test.shape}, y_test.shape={self.y_test.shape}')
        LOGGER.info(f'>>> Test Set dtype: X_test.dtype={self.X_test.dtype}, y_test.dtype={self.y_test.dtype}')

    def display_dataset_sample(self, num_samples: int = 9, cmap=plt.get_cmap('gray')) -> None:
        """ Displays random images from the dataset currently in memory. Maximum number of images to be displayed is
        min(100, batch size).

        Arguments:
            num_samples (int, optional): the number of images to be displayed
            cmap (Colormap, optional): the colormap to be used for displaying the images
        """
        # Parameter validation
        max_samples = min(self.X_train.shape[0], 100)
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

        # Plot random samples
        indices = [randrange(0, self.X_train.shape[0]) for _ in range(num_samples)]
        indices.extend([-1] * (grid_size ** 2 - num_samples))  # Pad with -1 for empty spaces

        _, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        for ax, i in zip(axes.flat, indices):
            if i == -1:
                ax.axis('off')
            else:
                sample = self.X_train[i].permute(1, 2, 0)  # Image must be channels-last in matplotlib
                label = self.class_labels[self.y_train[i]]
                ax.imshow(sample, cmap=cmap)
                ax.set_title(label)
                ax.axis('off')

        plt.show()

    def display_model(self) -> None:
        """ Logs information about the model currently in memory. """
        if self.model is not None:
            # Pass the channel size as 3 when fine tuning a classifier
            # pretrained on 3-channel images, on a grayscale dataset
            input_shape = self.dataset_shape
            if self.dataset_type == ClassifierDataset.FASHION_MNIST and self.X_train.shape[1] == 3:
                input_shape = (3, self.dataset_shape[1], self.dataset_shape[2])

            LOGGER.info('>>> Network components:')
            LOGGER.info(self.model)
            LOGGER.info('>>> Torchinfo summary:')
            summary(
                self.model,
                input_size=(1, *input_shape),
                col_names=["input_size", "output_size", "num_params",
                           "params_percent", "kernel_size", "mult_adds", "trainable"],
                verbose=1
            )
        else:
            LOGGER.info('>>> There is currently no model for this classifier')

    def save_results(self) -> None:
        """ Saves the current training run results by plotting training and validation accuracy and loss,
        and generating the classification report and confusion matrix. """
        self._create_current_run_directory()

        # Generate a file containing model information and parameters
        training_info_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'Training Information.txt'
        with open(training_info_path, 'w') as f:
            f.write('MODEL ARCHITECTURE:\n')
            f.write('------------------------------\n')
            for line in str(self.model).split('\n'):
                f.write(f'{line}\n')

            f.write('\nOPTIMIZER:\n')
            f.write('------------------------------\n')
            for line in str(self.optimizer).split('\n'):
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

        # Plot the train and validation accuracy and loss
        utils.plot_results(self.results_subdirectory, self.training_history)

        # Save the train and validation sets results
        results_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'Training Results.txt'
        with open(results_path, 'w') as f:
            f.write(json.dumps(self.training_history, indent=4))

        # Save the testing set results
        results_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'Testing Results.txt'
        with open(results_path, 'w') as f:
            f.write(json.dumps(self.evaluation_results, indent=4))

        # Generate the classification report
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(self.X_test)
            y_pred = torch.argmax(predictions, dim=1)

        report = sk_metrics.classification_report(self.y_test, y_pred, target_names=self.class_labels)
        report_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'Classification Report.txt'
        with open(report_path, 'w') as f:
            f.write(report)

        # Generate the confusion matrix
        cm = sk_metrics.confusion_matrix(self.y_test, y_pred, labels=list(range(len(self.class_labels))))
        utils.plot_confusion_matrix(cm, self.results_subdirectory, self.class_labels)

    def export_model(self) -> None:
        """ Exports the model currently in memory in ONNX format. """
        classifier_artifacts_path = config.CLASSIFIERS_PATH / self.results_subdirectory
        classifier_artifacts_path.mkdir()
        model_path = classifier_artifacts_path / 'model.onnx'
        dummy_input = torch.randn(1, *self.dataset_shape)
        self.model.eval()
        onnx_program = torch.onnx.dynamo_export(self.model, dummy_input)
        onnx_program.save(str(model_path))

    def _create_current_run_directory(self) -> None:
        """ Computes the run index of the current classifier training, creates a directory for the corresponding
        results and sets the name of the created directory as a class field. """
        current_run_cls_dataset = f"{self.__class__.__name__} {self.dataset_type.name}"
        training_runs = filter(lambda path: path.is_dir(), config.CLASSIFIER_RESULTS_PATH.iterdir())
        training_runs = list(map(lambda path: path.stem, training_runs))
        relevant_runs = list(filter(lambda name: name.startswith(current_run_cls_dataset), training_runs))

        if len(relevant_runs) == 0:
            current_run_dir_name = f"{current_run_cls_dataset} Run 1"
        else:
            run_numbers = [name.split(" ")[-1] for name in relevant_runs]
            latest_run = max(list(map(int, run_numbers)))
            current_run_dir_name = f"{current_run_cls_dataset} Run {latest_run + 1}"

        current_run_dir_path = config.CLASSIFIER_RESULTS_PATH / current_run_dir_name
        current_run_dir_path.mkdir()
        self.results_subdirectory = current_run_dir_name
