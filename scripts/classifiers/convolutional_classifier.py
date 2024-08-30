from copy import copy
from typing import Dict, List

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from tqdm import tqdm

from scripts import config, utils
from scripts.classifiers import TorchVisionDatasetClassifier
from scripts.models.cnn import CNN

LOGGER = utils.get_logger(__name__)


class CNNClassifier(TorchVisionDatasetClassifier):
    """ Class representing a classifier for Torchvision datasets, using a convolutional neural network """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and scaling the values. """
        if not self.preprocessed:
            if len(self.X_train.shape) == 3:
                # If the loaded dataset is grayscale, add the channel dimension
                self.X_train = torch.unsqueeze(self.X_train, dim=1)
                self.X_valid = torch.unsqueeze(self.X_valid, dim=1)
                self.X_test = torch.unsqueeze(self.X_test, dim=1)
            elif len(self.X_train.shape) == 4 and self.X_train.shape[3] == 3:
                # If the loaded dataset is RGB channels-last, transform to channel-first
                self.X_train = self.X_train.permute(0, 3, 1, 2)
                self.X_valid = self.X_valid.permute(0, 3, 1, 2)
                self.X_test = self.X_test.permute(0, 3, 1, 2)

            # Convert to float and scale to [0, 1]
            self.X_train = self.X_train.to(torch.float32) / 255.0
            self.X_valid = self.X_valid.to(torch.float32) / 255.0
            self.X_test = self.X_test.to(torch.float32) / 255.0

            self.preprocessed = True

    def build_model(self, compute_batch_size: bool = False) -> None:
        """ Defines the classifier's model structure and stores it as an instance attribute. The model used here is a
        convolutional neural network, consisting of 3 convolutional blocks (with pooling, dropout and L2
        regularization), followed by a decoder block (composed of 2 linear layers with dropout).

        Arguments:
            compute_batch_size (bool, optional): whether to compute the maximum batch size for this model and device
        """
        self.model = CNN(self.dataset_type).to(self.device)
        self.hyperparams = copy(config.CONVOLUTIONAL_CLF_HYPERPARAMS)

        if compute_batch_size and self.device is torch.device('cuda'):
            LOGGER.info('>>> Searching for the optimal batch size for this GPU...')
            dataset_size = sum(map(len, [self.X_train, self.X_valid, self.X_test]))
            self.hyperparams['BATCH_SIZE'] = utils.get_maximum_batch_size(self.model, self.device,
                                                                          input_shape=self.X_train.shape[1:],
                                                                          output_shape=self.y_train.shape[1:],
                                                                          dataset_size=dataset_size)

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
                                          lr=self.hyperparams['LEARNING_RATE'],
                                          weight_decay=0.0001 / self.hyperparams['NUM_EPOCHS'])
        self.loss = nn.CrossEntropyLoss()

        self._create_current_run_directory()

        # Define the evaluation metrics
        train_accuracy = MulticlassAccuracy(num_classes=len(self.class_labels), device=self.device)
        valid_accuracy = MulticlassAccuracy(num_classes=len(self.class_labels), device=self.device)
        train_precision = MulticlassPrecision(num_classes=len(self.class_labels), average='macro', device=self.device)
        valid_precision = MulticlassPrecision(num_classes=len(self.class_labels), average='macro', device=self.device)
        train_recall = MulticlassRecall(num_classes=len(self.class_labels), average='macro', device=self.device)
        valid_recall = MulticlassRecall(num_classes=len(self.class_labels), average='macro', device=self.device)
        train_f1 = MulticlassF1Score(num_classes=len(self.class_labels), average='macro', device=self.device)
        valid_f1 = MulticlassF1Score(num_classes=len(self.class_labels), average='macro', device=self.device)

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
        mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
        experiment_name = f"{self.__class__.__name__} {self.dataset_type.name}"
        mlflow.set_experiment(experiment_name)
        run_name = ' '.join(str(s) for s in self.results_subdirectory.split(' ')[2:])

        with mlflow.start_run(run_name=run_name, description=run_description, log_system_metrics=True) as run:
            # Define train and validation DataLoaders
            train_dataset = TensorDataset(self.X_train, self.y_train)
            valid_dataset = TensorDataset(self.X_valid, self.y_valid)
            train_dataloader = DataLoader(dataset=train_dataset,
                                          batch_size=self.hyperparams['BATCH_SIZE'],
                                          shuffle=True,
                                          pin_memory=self.pin_memory,
                                          pin_memory_device=self.pin_memory_device)
            valid_dataloader = DataLoader(dataset=valid_dataset,
                                          batch_size=self.hyperparams['BATCH_SIZE'],
                                          pin_memory=self.pin_memory,
                                          pin_memory_device=self.pin_memory_device)

            # Log the hyperparameters to MLflow
            mlflow.log_params(self.hyperparams)

            # Run the training loop
            for epoch in range(1, self.hyperparams['NUM_EPOCHS'] + 1):
                train_loss = 0.0
                self.model.train()

                for X_batch, y_batch in tqdm(train_dataloader):
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    self.optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    batch_loss = self.loss(y_pred, y_batch)
                    batch_loss.backward()
                    self.optimizer.step()

                    train_accuracy.update(y_pred, y_batch)
                    train_precision.update(y_pred, y_batch)
                    train_recall.update(y_pred, y_batch)
                    train_f1.update(y_pred, y_batch)

                    train_loss += batch_loss.item() / len(self.X_train)

                self.training_history["loss"].append(train_loss)
                self.training_history["accuracy"].append(train_accuracy.compute().item())
                self.training_history["precision"].append(train_precision.compute().item())
                self.training_history["recall"].append(train_recall.compute().item())
                self.training_history["f1-score"].append(train_f1.compute().item())

                # Log the training metrics to MLflow
                mlflow.log_metric('train_loss', self.training_history['loss'][-1], step=epoch)
                mlflow.log_metric('train_accuracy', self.training_history['accuracy'][-1], step=epoch)
                mlflow.log_metric('train_precision', self.training_history['precision'][-1], step=epoch)
                mlflow.log_metric('train_recall', self.training_history['recall'][-1], step=epoch)
                mlflow.log_metric('train_f1-score', self.training_history['f1-score'][-1], step=epoch)

                # Gradient computation is not required during the validation stage
                with torch.no_grad():
                    valid_loss = 0.0
                    self.model.eval()

                    for X_batch, y_batch in tqdm(valid_dataloader):
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self.model(X_batch)
                        batch_loss = self.loss(y_pred, y_batch)

                        valid_accuracy.update(y_pred, y_batch)
                        valid_precision.update(y_pred, y_batch)
                        valid_recall.update(y_pred, y_batch)
                        valid_f1.update(y_pred, y_batch)

                        valid_loss += batch_loss.item() / len(self.X_valid)

                    self.training_history["val_loss"].append(valid_loss)
                    self.training_history["val_accuracy"].append(valid_accuracy.compute().item())
                    self.training_history["val_precision"].append(valid_precision.compute().item())
                    self.training_history["val_recall"].append(valid_recall.compute().item())
                    self.training_history["val_f1-score"].append(valid_f1.compute().item())

                    # Log the validation metrics to MLflow
                    mlflow.log_metric('val_loss', self.training_history['val_loss'][-1], step=epoch)
                    mlflow.log_metric('val_accuracy', self.training_history['val_accuracy'][-1], step=epoch)
                    mlflow.log_metric('val_precision', self.training_history['val_precision'][-1], step=epoch)
                    mlflow.log_metric('val_recall', self.training_history['val_recall'][-1], step=epoch)
                    mlflow.log_metric('val_f1-score', self.training_history['val_f1-score'][-1], step=epoch)

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

                    if early_stopping_counter == self.hyperparams['EARLY_STOPPING_TOLERANCE']:
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

            self.run_id = run.info.run_id

        LOGGER.info(self.training_history)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by running it on the testing set. """
        # Define the loss function and evaluation metrics
        loss = nn.CrossEntropyLoss()
        accuracy = MulticlassAccuracy(num_classes=len(self.class_labels), device=self.device)
        precision = MulticlassPrecision(num_classes=len(self.class_labels), average='macro', device=self.device)
        recall = MulticlassRecall(num_classes=len(self.class_labels), average='macro', device=self.device)
        f1_score = MulticlassF1Score(num_classes=len(self.class_labels), average='macro', device=self.device)

        self.evaluation_results: Dict[str, float] = {
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1-score": 0.0,
        }

        # Setup and start an MLflow run
        mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
        experiment_name = f"{self.__class__.__name__} {self.dataset_type.name}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_id=self.run_id, log_system_metrics=True):
            # Define test DataLoader
            test_dataset = TensorDataset(self.X_test, self.y_test)
            test_dataloader = DataLoader(dataset=test_dataset,
                                         batch_size=self.hyperparams['BATCH_SIZE'],
                                         pin_memory=self.pin_memory,
                                         pin_memory_device=self.pin_memory_device)

            # Gradient computation is not required during evaluation
            with torch.no_grad():
                test_loss = 0.0
                self.model.eval()

                for X_batch, y_batch in tqdm(test_dataloader):
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    batch_loss = loss(y_pred, y_batch)

                    accuracy.update(y_pred, y_batch)
                    precision.update(y_pred, y_batch)
                    recall.update(y_pred, y_batch)
                    f1_score.update(y_pred, y_batch)

                    test_loss += batch_loss.item() / len(self.X_test)

                self.evaluation_results["test_loss"] = test_loss
                self.evaluation_results["test_accuracy"] = accuracy.compute().item()
                self.evaluation_results["test_precision"] = precision.compute().item()
                self.evaluation_results["test_recall"] = recall.compute().item()
                self.evaluation_results["test_f1-score"] = f1_score.compute().item()

                # Log the testing metrics to MLflow
                mlflow.log_metrics(self.evaluation_results)

        LOGGER.info(self.evaluation_results)
