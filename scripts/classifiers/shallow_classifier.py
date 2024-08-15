from copy import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from tqdm import tqdm

from scripts import config, utils
from scripts.classifiers import TorchVisionDatasetClassifier
from scripts.models.snn import SNN

LOGGER = utils.get_logger(__name__)


class SNNClassifier(TorchVisionDatasetClassifier):
    """ Class representing a classifier for Torchvision datasets, using a shallow neural network """
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

    def build_model(self) -> None:
        """ Defines the classifier's model structure and stores it as an instance attribute. The model used here is a
        shallow neural network, consisting only of the Input and Output layers. """
        self.model = SNN(self.dataset_type)
        self.hyperparams = copy(config.SHALLOW_CLF_HYPERPARAMS)

    def train_model(self) -> None:
        """ Defines the training parameters and runs the training loop for the model currently in memory. Vanilla SGD
        is used as the optimizer, the loss function to be optimised is the Categorical Cross-entropy loss, and the
        measured metrics are Accuracy (which is appropriate for our problem, because the dataset classes are balanced),
        Precision, Recall, and F1-Score. An early stopping mechanism is used to prevent overfitting. """
        # Define the optimizer and loss functions
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyperparams['LEARNING_RATE'])
        self.loss = nn.CrossEntropyLoss()

        # Define the evaluation metrics
        train_accuracy = MulticlassAccuracy(num_classes=len(self.class_labels))
        valid_accuracy = MulticlassAccuracy(num_classes=len(self.class_labels))
        train_precision = MulticlassPrecision(num_classes=len(self.class_labels), average='macro')
        valid_precision = MulticlassPrecision(num_classes=len(self.class_labels), average='macro')
        train_recall = MulticlassRecall(num_classes=len(self.class_labels), average='macro')
        valid_recall = MulticlassRecall(num_classes=len(self.class_labels), average='macro')
        train_f1 = MulticlassF1Score(num_classes=len(self.class_labels), average='macro')
        valid_f1 = MulticlassF1Score(num_classes=len(self.class_labels), average='macro')

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

        # Define train and validation DataLoaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        valid_dataset = TensorDataset(self.X_valid, self.y_valid)
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=self.hyperparams['BATCH_SIZE'],
                                      shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=self.hyperparams['BATCH_SIZE'])

        # Run the training loop
        for epoch in range(1, self.hyperparams['NUM_EPOCHS'] + 1):
            train_loss = 0.0
            self.model.train()

            for X_batch, y_batch in tqdm(train_dataloader):
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

            # Gradient computation is not required during the validation stage
            with torch.no_grad():
                valid_loss = 0.0
                self.model.eval()

                for X_batch, y_batch in tqdm(valid_dataloader):
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

        LOGGER.info(self.training_history)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by running it on the testing set. """
        # Define the loss function and evaluation metrics
        loss = nn.CrossEntropyLoss()
        accuracy = MulticlassAccuracy(num_classes=len(self.class_labels))
        precision = MulticlassPrecision(num_classes=len(self.class_labels), average='macro')
        recall = MulticlassRecall(num_classes=len(self.class_labels), average='macro')
        f1_score = MulticlassF1Score(num_classes=len(self.class_labels), average='macro')

        self.evaluation_results: Dict[str, float] = {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        }

        # Define test DataLoader
        test_dataset = TensorDataset(self.X_test, self.y_test)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=self.hyperparams['BATCH_SIZE'])

        # Gradient computation is not required during evaluation
        with torch.no_grad():
            test_loss = 0.0
            self.model.eval()

            for X_batch, y_batch in tqdm(test_dataloader):
                y_pred = self.model(X_batch)
                batch_loss = loss(y_pred, y_batch)

                accuracy.update(y_pred, y_batch)
                precision.update(y_pred, y_batch)
                recall.update(y_pred, y_batch)
                f1_score.update(y_pred, y_batch)

                test_loss += batch_loss.item() / len(self.X_test)

            self.evaluation_results["loss"] = test_loss
            self.evaluation_results["accuracy"] = accuracy.compute().item()
            self.evaluation_results["precision"] = precision.compute().item()
            self.evaluation_results["recall"] = recall.compute().item()
            self.evaluation_results["f1-score"] = f1_score.compute().item()

            LOGGER.info(self.evaluation_results)
