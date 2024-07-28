import json
from abc import ABC
from math import sqrt
from random import randrange
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk_metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.optimizers import serialize
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflowjs.converters import save_keras_model

import torch
from torchinfo import summary
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision.transforms import ToTensor

from scripts import config, utils
from scripts.config import DatasetType
from scripts.interfaces import FashionMNISTModel


LOGGER = utils.get_logger(__name__)


class FashionMNISTClassifier(FashionMNISTModel, ABC):
    """ Abstract class representing the blueprint all classifiers on TorchVision datasets must follow """
    def __init__(self, dataset: DatasetType) -> None:
        """ Loads the specified dataset and stores it in instance attributes """
        self.dataset = dataset

        match self.dataset:
            case config.DatasetType.FASHION_MNIST:
                train_dataset = FashionMNIST(root='data', train=True, download=True)
                test_dataset = FashionMNIST(root='data', train=False, download=True)
            case config.DatasetType.CIFAR_10:
                train_dataset = CIFAR10(root='data', train=True, download=True)
                test_dataset = CIFAR10(root='data', train=False, download=True)

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
        self.results_subdirectory = None

    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, 'preprocess_dataset') and callable(subclass.preprocess_dataset) and
                hasattr(subclass, 'build_model') and callable(subclass.build_model) and
                hasattr(subclass, 'train_model') and callable(subclass.train_model) and
                hasattr(subclass, 'evaluate_model') and callable(subclass.evaluate_model))

    def display_dataset_information(self) -> None:
        """ Logs information about the dataset currently in memory """
        LOGGER.info(f'>>> Train Set Shape: X_train.shape={self.X_train.shape}, y_train.shape={self.y_train.shape}')
        LOGGER.info(f'>>> Validation Set Shape: X_valid.shape={self.X_valid.shape}, y_valid.shape={self.y_valid.shape}')
        LOGGER.info(f'>>> Test Set Shape: X_test.shape={self.X_test.shape}, y_test.shape={self.y_test.shape}')

    def display_dataset_sample(self, num_samples: int = 9, cmap=plt.get_cmap('gray')) -> None:
        """ Displays some images from the dataset currently in memory.
        Maximum number of images to be displayed is min(100, batch size)

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
        idx = 1

        for i in indices:
            plt.subplot(grid_size, grid_size, idx)
            plt.imshow(self.X_train[i], cmap=cmap)
            plt.axis('off')
            idx += 1

        plt.show()

    def display_model(self) -> None:
        """ Logs information about the model currently in memory """
        if self.model is not None:
            # self.model.summary()
            LOGGER.info(str(self.model))
            LOGGER.info(
                summary(
                    self.model,
                    # input_size=(
                    #     constants.BATCH_SIZE,
                    #     constants.NUM_CHANNELS,
                    #     constants.IMAGE_HEIGHT,
                    #     constants.IMAGE_WIDTH,
                    # ),
                    # col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"],
                    # verbose=1
                )
            )
        else:
            LOGGER.info('>>> There is currently no model for this classifier')

    def save_results(
        self,
        hyperparams: Dict[str, int],
        training_results: Dict[str, List[float]],
        testing_results: Dict[str, float]
    ) -> None:
        """ Evaluates the model currently in memory by plotting training and validation
        accuracy and loss and generating the classification report and confusion matrix

        Arguments:
            results_dir (Path): the path to where the training results will be saved

            training_results (Dict[str, List[float]]): dictionary containing the loss values and
                the accuracy, precision, recall and F1 score results, both on the training and
                validation sets

            test_results (Dict[str, float]): dictionary containing the loss, accuracy, precision,
                recall and F1 score results on the test set
        """
        self._create_current_run_directory()

        # Plot the train and validation accuracy and loss
        utils.plot_results(self.results_subdirectory, training_results)

        # Save the train and validation sets results
        results_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'Training Results.txt'
        with open(results_path, 'w') as f:
            f.write(json.dumps(training_results, indent=4))

        # Save the testing set results
        results_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'Testing Results.txt'
        with open(results_path, 'w') as f:
            f.write(json.dumps(testing_results, indent=4))

        # Generate the classification report
        predictions = self.model.predict(x=self.X_test, batch_size=hyperparams['BATCH_SIZE'], verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_pred_categorical = to_categorical(y_pred)

        report = sk_metrics.classification_report(self.y_test, y_pred_categorical, target_names=config.CLASS_LABELS)
        report_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'Classification Report.txt'
        with open(report_path, 'w') as f:
            f.write(report)

        # Generate the confusion matrix
        self.y_test = np.argmax(self.y_test, axis=1)
        cm = sk_metrics.confusion_matrix(self.y_test, y_pred, labels=list(range(10)))
        utils.plot_confusion_matrix(cm, self.results_subdirectory, config.CLASS_LABELS)

        # Generate a file containing model information and parameters
        training_info_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'Training Information.txt'
        try:
            model_str = json.loads(self.model.to_json())
        except NotImplementedError:
            model_str = 'N/A'
        training_info = {
            'batch_size': hyperparams['BATCH_SIZE'],
            'num_epochs': hyperparams['NUM_EPOCHS'],
            'model': model_str,
            'optimizer': str(serialize(self.model.optimizer))
        }
        with open(training_info_path, 'w') as f:
            f.write(json.dumps(training_info, indent=4))

    def export_model(self) -> None:
        """ Exports the model currently in memory in Tensorflow.js format """
        artifacts_path = config.CLASSIFIERS_PATH / self.results_subdirectory
        save_keras_model(self.model, artifacts_path)

    def _create_current_run_directory(self) -> None:
        """ Computes the run index of the current classifier training, creates a directory for the corresponding
        results and sets the name of the created directory as a class field """
        training_runs = filter(lambda path: path.is_dir(), config.CLASSIFIER_RESULTS_PATH.iterdir())
        training_runs = list(map(lambda path: path.stem, training_runs))
        relevant_runs = list(filter(lambda name: name.startswith(self.__class__.__name__), training_runs))

        if len(relevant_runs) == 0:
            current_run_dir_name = f"{self.__class__.__name__} Run 1"
        else:
            run_numbers = [name.split(" ")[-1] for name in relevant_runs]
            latest_run = max(list(map(int, run_numbers)))
            current_run_dir_name = f"{self.__class__.__name__} Run {latest_run + 1}"

        current_run_dir_path = config.CLASSIFIER_RESULTS_PATH / current_run_dir_name
        current_run_dir_path.mkdir()
        self.results_subdirectory = current_run_dir_name
