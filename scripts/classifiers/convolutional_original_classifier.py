from typing import Dict

import tensorflow.python.keras.regularizers as regularizers
from tensorflow.python.keras.activations import relu, softmax
from tensorflow.python.keras.callbacks import EarlyStopping, History, TensorBoard
from tensorflow.python.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
    MaxPooling2D,
)
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from scripts import config, utils
from scripts.classifiers import FashionMNISTClassifier

LOGGER = utils.get_logger(__name__)


class CNNOriginalClassifier(FashionMNISTClassifier):
    """ Class representing a strong classifier for the original Fashion-MNIST dataset, using a convolutional neural
     network """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and encoding the labels """
        if not self.preprocessed:
            self.X_train = self.X_train.reshape((*self.X_train.shape, 1)).astype(float) / 255.0
            self.X_valid = self.X_valid.reshape((*self.X_valid.shape, 1)).astype(float) / 255.0
            self.X_test = self.X_test.reshape((*self.X_test.shape, 1)).astype(float) / 255.0

            self.y_train = to_categorical(self.y_train)
            self.y_valid = to_categorical(self.y_valid)
            self.y_test = to_categorical(self.y_test)

            self.preprocessed = True

    def build_model(self) -> None:
        """ Defines the classifier model structure and stores it as an instance attribute. The model used here is a
         convolutional neural network, consisting of 3 convolutional blocks with pooling, dropout and L2 regularization,
         followed by 2 dense layers with dropout and Adam as optimizer. Early stopping and TensorBoard callbacks are
         also implemented """
        l2 = regularizers.l2(config.L2_LOSS_LAMBDA_2)

        self.model = Sequential(name='CNNOriginalClassifier')
        self.model.add(InputLayer(input_shape=(self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]),
                                  dtype=float,
                                  name='original_image'))

        self.model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(rate=0.2))

        self.model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(rate=0.3))

        self.model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(rate=0.4))

        self.model.add(Flatten())

        self.model.add(Dense(units=512, activation=relu, kernel_initializer='he_uniform'))
        self.model.add(Dropout(rate=0.2))

        self.model.add(Dense(10, activation=softmax, kernel_initializer='he_uniform'))

        optimizer = Adam(learning_rate=0.0001, decay=0.0001 / config.CONVOLUTIONAL_CLF_HYPERPARAMS['NUM_EPOCHS'])
        self.model.compile(optimizer=optimizer,
                           loss=CategoricalCrossentropy(),
                           metrics=[CategoricalAccuracy(), Precision(), Recall()])

    def train_model(self) -> None:
        """ Performs the training and evaluation of this classifier, on both the train set and the validation set.
         The loss function to be optimised is the Categorical Cross-entropy loss and the measured metric is Accuracy,
          which is appropriate for our problem, because the dataset classes are balanced.  """
        self._create_current_run_directory()
        logs_path = config.CLASSIFIER_RESULTS_PATH / self.results_subdirectory / 'logs'
        es_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        tb_callback = TensorBoard(log_dir=logs_path)

        self.training_history = self.model.fit(x=self.X_train, y=self.y_train,
                                               batch_size=config.CONVOLUTIONAL_CLF_HYPERPARAMS['BATCH_SIZE'],
                                               epochs=config.CONVOLUTIONAL_CLF_HYPERPARAMS['NUM_EPOCHS'],
                                               verbose=1, callbacks=[es_callback, tb_callback],
                                               validation_data=(self.X_valid, self.y_valid)).history
        self.test_accuracy = self.model.evaluate(x=self.X_test, y=self.y_test,
                                                 batch_size=config.CONVOLUTIONAL_CLF_HYPERPARAMS['BATCH_SIZE'],
                                                 verbose=1, return_dict=True)

    def evaluate_model(self, training_history: History, test_accuracy: Dict[str, float]) -> None:
        """ Evaluates the model currently in memory by plotting training and validation accuracy and loss and generating
        the classification report and confusion matrix """
        super().evaluate_model(config.CONVOLUTIONAL_CLF_HYPERPARAMS, training_history, test_accuracy)
