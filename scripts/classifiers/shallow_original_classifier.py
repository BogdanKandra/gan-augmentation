from typing import Dict, List

from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.callbacks import EarlyStopping, History
from tensorflow.python.keras.layers import Dense, Flatten, InputLayer
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical

import torch
import torch.nn.functional as F

from scripts import config, utils
from scripts.classifiers import FashionMNISTClassifier
from scripts.models.snn import SNN

LOGGER = utils.get_logger(__name__)


class SNNOriginalClassifier(FashionMNISTClassifier):
    """ Class representing a weak classifier for the original Fashion-MNIST dataset, using a shallow neural network """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and encoding the labels """
        if not self.preprocessed:
            # If the loaded dataset is grayscale, add the channel dimension
            if len(self.X_train.shape) == 3:
                self.X_train = torch.unsqueeze(self.X_train, dim=3)
                self.X_valid = torch.unsqueeze(self.X_valid, dim=3)
                self.X_test = torch.unsqueeze(self.X_test, dim=3)

            # Convert to float and rescale to [0, 1]
            self.X_train = self.X_train.to(float) / 255.0
            self.X_valid = self.X_valid.to(float) / 255.0
            self.X_test = self.X_test.to(float) / 255.0

            # Encode the labels as one-hot vectors
            dataset_class_labels = getattr(config, self.dataset_type.name + '_CLASS_LABELS')
            self.y_train = F.one_hot(self.y_train, num_classes=len(dataset_class_labels)).to(float)
            self.y_valid = F.one_hot(self.y_valid, num_classes=len(dataset_class_labels)).to(float)
            self.y_test = F.one_hot(self.y_test, num_classes=len(dataset_class_labels)).to(float)

            self.preprocessed = True

    def build_model(self) -> None:
        """ Defines the classifier model structure and stores it as an instance attribute. The model used here is a
         shallow neural network, consisting only of the Input and Output layers, with a vanilla SGD as optimizer """
        self.model = SNN(self.dataset_type)

        # self.model = Sequential(name='SNNOriginalClassifier')
        # self.model.add(InputLayer(input_shape=(self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]),
        #                           dtype=float,
        #                           name='original_image'))
        # self.model.add(Flatten())
        # self.model.add(Dense(units=10, activation=softmax, kernel_initializer='he_uniform'))
        # optimizer = SGD(learning_rate=0.01)
        # self.model.compile(optimizer=optimizer,
        #                    loss=CategoricalCrossentropy(),
        #                    metrics=[CategoricalAccuracy(), Precision(), Recall()])

    def train_model(self) -> Dict[str, List[float]]:
        """ Defines the training parameters and runs the training loop for the model currently in memory.
        The loss function to be optimised is the Categorical Cross-entropy loss and the measured metric is Accuracy,
        which is appropriate for our problem, because the dataset classes are balanced.

        Returns:
            Dict[str, List[float]]: dictionary containing the loss values and the accuracy,
                precision, recall and F1 score results, both on the training and validation sets
        """
        es_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

        self.training_history = self.model.fit(x=self.X_train, y=self.y_train,
                                               batch_size=config.SHALLOW_CLF_HYPERPARAMS['BATCH_SIZE'],
                                               epochs=config.SHALLOW_CLF_HYPERPARAMS['NUM_EPOCHS'],
                                               verbose=1, callbacks=[es_callback],
                                               validation_data=(self.X_valid, self.y_valid)).history

    def evaluate_model(self) -> Dict[str, float]:
        """ Evaluates the model currently in memory by running it on the testing set.
        
        Returns:
            Dict[str, float]: dictionary containing the loss value and the accuracy,
                precision, recall and F1 score results on the testing set
        """
        self.test_accuracy = self.model.evaluate(x=self.X_test, y=self.y_test,
                                                 batch_size=config.SHALLOW_CLF_HYPERPARAMS['BATCH_SIZE'],
                                                 verbose=1, return_dict=True)

    def save_results(self, training_history: History, test_accuracy: Dict[str, float]) -> None:
        """ Evaluates the model currently in memory by plotting training and validation accuracy and loss
        and generating the classification report and confusion matrix """
        super().save_results(config.SHALLOW_CLF_HYPERPARAMS, training_history, test_accuracy)
