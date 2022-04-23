from scripts.classifiers.fashion_mnist_classifier import FashionMNISTClassifier
from scripts import config, utils
import numpy as np
from keras.activations import relu, softmax
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


LOGGER = utils.get_logger(__name__)


class StrongOriginalClassifier(FashionMNISTClassifier):
    """ Class representing a strong classifier for the original Fashion-MNIST dataset """
    def __init__(self):
        """ This just calls the base class' constructor """
        super().__init__()

    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and encoding the labels """
        self.X_train = self.X_train.reshape((*self.X_train.shape, 1)).astype(float) / 255.0
        self.X_test = self.X_test.reshape((*self.X_test.shape, 1)).astype(float) / 255.0

        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def build_model(self) -> None:
        """ Defines the classifier model structure and stores it as an instance attribute """
        self.model = Sequential(name='StrongOriginalClassifier')
        self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]),
                             name='original_image',
                             dtype=float))
        self.model.add(Conv2D(filters=32, kernel_size=3, activation=relu, kernel_initializer='he_uniform'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Flatten())
        # self.model.add(Dense(units=50, activation=relu, kernel_initializer='he_uniform'))
        self.model.add(Dense(10, activation=softmax, kernel_initializer='he_uniform'))
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(optimizer=optimizer,
                           loss=CategoricalCrossentropy(),
                           metrics=[CategoricalAccuracy()])

    def train_model(self) -> None:
        """ Performs the training and evaluation of this classifier, on both the train set and the validation set.
         The loss function to be optimised is the Categorical Cross-entropy loss and the measured metric is Accuracy,
          which is appropriate for our problem, because the dataset classes are balanced.  """
        self.__training_history = self.model.fit(x=self.X_train, y=self.y_train, batch_size=config.BATCH_SIZE,
                                                 epochs=config.NUM_EPOCHS, verbose=1,
                                                 validation_data=(self.X_valid, self.y_valid))
        self.__test_accuracy = self.model.evaluate(x=self.X_test, y=self.y_test, batch_size=config.BATCH_SIZE,
                                                   verbose=1, return_dict=True)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by plotting training and validation accuracy and loss and generating
        the classification report and confusion matrix """


if __name__ == '__main__':
    clf = StrongOriginalClassifier()
    clf.preprocess_dataset()
    clf.build_model()
    clf.display_model()













