from copy import copy
from typing import Dict, List

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision import transforms
from tqdm import tqdm

from scripts import config, utils
from scripts.classifiers import AbstractClassifier
from scripts.config import ClassifierDataset
from scripts.models.efficient_net import EfficientNet

LOGGER = utils.get_logger(__name__)


class EfficientNet_Classifier(AbstractClassifier):
    """ Class representing a classifier for Torchvision datasets, using the transfer learning approach """
    def preprocess_dataset(self) -> None:
        """ Loads the specified dataset and preprocesses it. The data is converted to channels-first torch.FloatTensor,
        scaled to the [0.0, 1.0] range, resized to the size expected by the EfficientNet pre-trained network, and
        normalized. The labels are one-hot encoded. If the dataset is grayscale, the channel dimension is squeezed in.
        The preprocessing is only applied when iterating over the dataset with a DataLoader. """
        if not self.preprocessed:
            self.hyperparams = copy(config.EFFICIENTNET_CLF_HYPERPARAMS)

            # Set the preprocessing transforms
            grayscale_transforms = transforms.Compose([
                transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            color_transforms = transforms.Compose([
                transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Normalize the images with the dataset mean and standard deviation
            # mean = torch.mean(self.X_train, dim=(0, 2, 3), keepdim=True)
            # std = torch.std(self.X_train, dim=(0, 2, 3), keepdim=True)

            # Load the dataset and apply the preprocessing transforms
            match self.dataset_type:
                case ClassifierDataset.FASHION_MNIST:
                    self.train_dataset = FashionMNIST(root="data",
                                                      train=True,
                                                      transform=grayscale_transforms,
                                                      download=True)
                    self.test_dataset = FashionMNIST(root="data",
                                                     train=False,
                                                     transform=grayscale_transforms,
                                                     download=True)
                case ClassifierDataset.CIFAR_10:
                    self.train_dataset = CIFAR10(root="data",
                                                 train=True,
                                                 transform=color_transforms,
                                                 download=True)
                    self.test_dataset = CIFAR10(root="data",
                                                train=False,
                                                transform=color_transforms,
                                                download=True)

            # Create samplers for randomly splitting the training dataset into a training and validation set
            train_set_size = len(self.train_dataset)
            train_set_indices = list(range(train_set_size))
            np.random.shuffle(train_set_indices)

            validation_split_index = int(train_set_size * config.VALID_SET_PERCENTAGE)
            train_idxs = train_set_indices[validation_split_index:]
            valid_idxs = train_set_indices[:validation_split_index]

            self.train_sampler = SubsetRandomSampler(train_idxs)
            self.valid_sampler = SubsetRandomSampler(valid_idxs)

            # Define train, validation and test DataLoaders
            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                               batch_size=self.hyperparams["BATCH_SIZE"],
                                               sampler=self.train_sampler,
                                               **self.dataloader_params)
            self.valid_dataloader = DataLoader(dataset=self.train_dataset,
                                               batch_size=self.hyperparams["BATCH_SIZE"],
                                               sampler=self.valid_sampler,
                                               **self.dataloader_params)
            self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                              batch_size=self.hyperparams["BATCH_SIZE"],
                                              **self.dataloader_params)

            # Store the shapes of the batch data and labels tensors
            batch, labels = next(iter(self.train_dataloader))
            self.batch_shape = batch.shape
            self.labels_shape = labels.shape

            self.preprocessed = True

    def build_model(self, compute_batch_size: bool = False) -> None:
        """ Defines the classifier's model structure and stores it as an instance attribute. The model used here is the
        headless EfficientNetB0 pretrained network, with a new classifier head consisting of Dropout and the Output
        layer.

        Arguments:
            compute_batch_size (bool, optional): whether to compute the maximum batch size for this model and device
        """
        self.model = EfficientNet(self.dataset_type).to(self.device, non_blocking=self.non_blocking)

        if compute_batch_size:
            if self.device.type == "cuda":
                LOGGER.info(">>> Searching for the optimal batch size for this model and GPU...")
                temp_model = EfficientNet(self.dataset_type).to(self.device, non_blocking=self.non_blocking)
                minimum_dataset_size = min(map(len, [self.train_sampler, self.valid_sampler, self.test_dataset.data]))
                optimal_batch_size = utils.get_maximum_classifier_batch_size(temp_model, self.device,
                                                                             input_shape=self.batch_shape[1:],
                                                                             output_shape=self.labels_shape[1:],
                                                                             dataset_size=minimum_dataset_size,
                                                                             max_batch_size=1024)
                self.hyperparams["BATCH_SIZE"] = optimal_batch_size
                del temp_model
            else:
                LOGGER.info(">>> GPU not available, batch size computation skipped.")

    def train_model(self, run_description: str) -> None:
        """ Defines the training parameters and runs the training loop for the model currently in memory. Adam is used
        as the optimizer, the loss function to be optimised is the Categorical Cross-entropy loss, and the measured
        metrics are Accuracy (which is appropriate for our problem, because the dataset classes are balanced),
        Precision, Recall, and F1-Score. An early stopping mechanism is used to prevent overfitting.

        Arguments:
            run_description (str): The description of the current run
        """
        # Define the optimizer and loss functions
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.hyperparams["LEARNING_RATE"],
                                          weight_decay=0.01 / self.hyperparams["NUM_EPOCHS"])
        self.loss = nn.CrossEntropyLoss()

        self._create_current_run_directory()

        # Define the evaluation metrics
        train_accuracy = MulticlassAccuracy(num_classes=len(self.class_labels), device=self.device)
        valid_accuracy = MulticlassAccuracy(num_classes=len(self.class_labels), device=self.device)
        train_precision = MulticlassPrecision(num_classes=len(self.class_labels), average="macro", device=self.device)
        valid_precision = MulticlassPrecision(num_classes=len(self.class_labels), average="macro", device=self.device)
        train_recall = MulticlassRecall(num_classes=len(self.class_labels), average="macro", device=self.device)
        valid_recall = MulticlassRecall(num_classes=len(self.class_labels), average="macro", device=self.device)
        train_f1 = MulticlassF1Score(num_classes=len(self.class_labels), average="macro", device=self.device)
        valid_f1 = MulticlassF1Score(num_classes=len(self.class_labels), average="macro", device=self.device)

        # Keep track of metrics for evaluation
        self.training_history: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
            "precision": [],
            "val_precision": [],
            "recall": [],
            "val_recall": [],
            "f1-score": [],
            "val_f1-score": [],
        }
        early_stopping_counter = 0

        # Setup and start an MLflow run
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        experiment_name = f"{self.__class__.__name__} {self.dataset_type.name}"
        mlflow.set_experiment(experiment_name)
        run_name = " ".join(str(s) for s in self.results_subdirectory.split(" ")[2:])

        with mlflow.start_run(run_name=run_name, description=run_description, log_system_metrics=True) as run:
            self.run_id = run.info.run_id

            # Log the hyperparameters to MLflow
            mlflow.log_params(self.hyperparams)

            # Run the training loop
            self.model.train()

            for epoch in range(1, self.hyperparams["NUM_EPOCHS"] + 1):
                train_loss = 0.0

                for X_batch, y_batch in tqdm(self.train_dataloader):
                    X_batch = X_batch.to(self.device, non_blocking=self.non_blocking)
                    y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)
                    self.optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    batch_loss = self.loss(y_pred, y_batch)
                    batch_loss.backward()
                    self.optimizer.step()

                    train_accuracy.update(y_pred, y_batch)
                    train_precision.update(y_pred, y_batch)
                    train_recall.update(y_pred, y_batch)
                    train_f1.update(y_pred, y_batch)

                    train_loss += batch_loss.item()

                self.training_history["loss"].append(train_loss)
                self.training_history["accuracy"].append(train_accuracy.compute().item())
                self.training_history["precision"].append(train_precision.compute().item())
                self.training_history["recall"].append(train_recall.compute().item())
                self.training_history["f1-score"].append(train_f1.compute().item())

                # Log the training metrics to MLflow
                mlflow.log_metric("train_loss", self.training_history["loss"][-1], step=epoch)
                mlflow.log_metric("train_accuracy", self.training_history["accuracy"][-1], step=epoch)
                mlflow.log_metric("train_precision", self.training_history["precision"][-1], step=epoch)
                mlflow.log_metric("train_recall", self.training_history["recall"][-1], step=epoch)
                mlflow.log_metric("train_f1-score", self.training_history["f1-score"][-1], step=epoch)

                # Gradient computation is not required during the validation stage
                with torch.no_grad():
                    valid_loss = 0.0
                    self.model.eval()

                    for X_batch, y_batch in tqdm(self.valid_dataloader):
                        X_batch = X_batch.to(self.device, non_blocking=self.non_blocking)
                        y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)
                        y_pred = self.model(X_batch)
                        batch_loss = self.loss(y_pred, y_batch)

                        valid_accuracy.update(y_pred, y_batch)
                        valid_precision.update(y_pred, y_batch)
                        valid_recall.update(y_pred, y_batch)
                        valid_f1.update(y_pred, y_batch)

                        valid_loss += batch_loss.item()

                    self.training_history["val_loss"].append(valid_loss)
                    self.training_history["val_accuracy"].append(valid_accuracy.compute().item())
                    self.training_history["val_precision"].append(valid_precision.compute().item())
                    self.training_history["val_recall"].append(valid_recall.compute().item())
                    self.training_history["val_f1-score"].append(valid_f1.compute().item())

                    # Log the validation metrics to MLflow
                    mlflow.log_metric("val_loss", self.training_history["val_loss"][-1], step=epoch)
                    mlflow.log_metric("val_accuracy", self.training_history["val_accuracy"][-1], step=epoch)
                    mlflow.log_metric("val_precision", self.training_history["val_precision"][-1], step=epoch)
                    mlflow.log_metric("val_recall", self.training_history["val_recall"][-1], step=epoch)
                    mlflow.log_metric("val_f1-score", self.training_history["val_f1-score"][-1], step=epoch)

                    curr_train_acc = self.training_history["accuracy"][-1]
                    curr_val_acc = self.training_history["val_accuracy"][-1]

                    LOGGER.info(f"Epoch: {epoch}/{self.hyperparams['NUM_EPOCHS']}")
                    LOGGER.info(f"> loss: {train_loss}\t val_loss: {valid_loss}")
                    LOGGER.info(f"> accuracy: {curr_train_acc}\t val_accuracy: {curr_val_acc}")

                    # Early stopping
                    best_loss = min(self.training_history["val_loss"])
                    if valid_loss <= best_loss:
                        if early_stopping_counter != 0:
                            early_stopping_counter = 0
                            LOGGER.info(">> Early stopping counter reset.")
                        self.best_model_state = self.model.state_dict()
                    else:
                        early_stopping_counter += 1
                        LOGGER.info(f">> Early stopping counter increased to {early_stopping_counter}.")

                    if early_stopping_counter == self.hyperparams["EARLY_STOPPING_TOLERANCE"]:
                        LOGGER.info(">> Training terminated due to early stopping!")
                        break

                train_accuracy.reset()
                train_precision.reset()
                train_recall.reset()
                train_f1.reset()
                valid_accuracy.reset()
                valid_precision.reset()
                valid_recall.reset()
                valid_f1.reset()

        LOGGER.info(self.training_history)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by running it on the testing set. """
        # Define the loss function and evaluation metrics
        loss = nn.CrossEntropyLoss()
        accuracy = MulticlassAccuracy(num_classes=len(self.class_labels), device=self.device)
        precision = MulticlassPrecision(num_classes=len(self.class_labels), average="macro", device=self.device)
        recall = MulticlassRecall(num_classes=len(self.class_labels), average="macro", device=self.device)
        f1_score = MulticlassF1Score(num_classes=len(self.class_labels), average="macro", device=self.device)

        self.evaluation_results: Dict[str, float] = {
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1-score": 0.0,
        }

        # Setup and start an MLflow run
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        experiment_name = f"{self.__class__.__name__} {self.dataset_type.name}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_id=self.run_id, log_system_metrics=True):
            # Gradient computation is not required during evaluation
            with torch.no_grad():
                test_loss = 0.0
                self.model.eval()

                for X_batch, y_batch in tqdm(self.test_dataloader):
                    X_batch = X_batch.to(self.device, non_blocking=self.non_blocking)
                    y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)
                    y_pred = self.model(X_batch)
                    batch_loss = loss(y_pred, y_batch)

                    accuracy.update(y_pred, y_batch)
                    precision.update(y_pred, y_batch)
                    recall.update(y_pred, y_batch)
                    f1_score.update(y_pred, y_batch)

                    test_loss += batch_loss.item()

                self.evaluation_results["test_loss"] = test_loss
                self.evaluation_results["test_accuracy"] = accuracy.compute().item()
                self.evaluation_results["test_precision"] = precision.compute().item()
                self.evaluation_results["test_recall"] = recall.compute().item()
                self.evaluation_results["test_f1-score"] = f1_score.compute().item()

                # Log the testing metrics to MLflow
                mlflow.log_metrics(self.evaluation_results)

        LOGGER.info(self.evaluation_results)
