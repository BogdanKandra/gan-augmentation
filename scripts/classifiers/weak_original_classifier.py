import json
import os
from scripts.classifiers.fashion_mnist_classifier import FashionMNISTClassifier
from scripts import config, utils
from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as sk_metrics
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


LOGGER = utils.get_logger(__name__)


class WeakOriginalClassifier(FashionMNISTClassifier):
    """ Class representing a weak classifier for the original Fashion-MNIST dataset """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and encoding the labels """
        self.X_train = self.X_train.reshape((*self.X_train.shape, 1)).astype(float) / 255.0
        self.X_valid = self.X_valid.reshape((*self.X_valid.shape, 1)).astype(float) / 255.0
        self.X_test = self.X_test.reshape((*self.X_test.shape, 1)).astype(float) / 255.0

        self.y_train = to_categorical(self.y_train)
        self.y_valid = to_categorical(self.y_valid)
        self.y_test = to_categorical(self.y_test)

    def build_model(self) -> None:
        """ Defines the classifier model structure and stores it as an instance attribute """
        self.model = Sequential(name='WeakOriginalClassifier')
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
        self.__training_history = self.model.fit(x=self.X_train, y=self.y_train, batch_size=config.BATCH_SIZE_WEAK,
                                                 epochs=config.NUM_EPOCHS_WEAK, verbose=1,
                                                 validation_data=(self.X_valid, self.y_valid)).history
        self.__test_accuracy = self.model.evaluate(x=self.X_test, y=self.y_test, batch_size=config.BATCH_SIZE_WEAK,
                                                   verbose=1, return_dict=True)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by plotting training and validation accuracy and loss and generating
        the classification report and confusion matrix """
        results_subdirectory_name = self.create_current_run_directory()
        utils.plot_results(results_subdirectory_name, self.__training_history)
        results_name = 'Training Results.txt'
        with open(os.path.join(config.CNN_RESULTS_PATH, results_subdirectory_name, results_name), 'w') as f:
            f.write(json.dumps(self.__training_history, indent=4))

        # Save the test results
        results_name = 'Test Results.txt'
        with open(os.path.join(config.CNN_RESULTS_PATH, results_subdirectory_name, results_name), 'w') as f:
            f.write(json.dumps(self.__test_accuracy, indent=4))

        # Generate the classification report
        predictions = self.model.predict(x=self.X_test, batch_size=config.BATCH_SIZE_WEAK, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_pred_categorical = to_categorical(y_pred)
        class_labels = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        report = sk_metrics.classification_report(self.y_test, y_pred_categorical, target_names=class_labels)
        report_name = 'Classification Report.txt'
        with open(os.path.join(config.CNN_RESULTS_PATH, results_subdirectory_name, report_name), 'w') as f:
            f.write(report)

        self.y_test = np.argmax(self.y_test, axis=1)
        cm = sk_metrics.confusion_matrix(self.y_test, y_pred, labels=list(range(10)))
        utils.plot_confusion_matrix(cm, results_subdirectory_name, class_labels)

if __name__ == '__main__':
    clf = WeakOriginalClassifier()
    clf.preprocess_dataset()
    clf.build_model()
    clf.display_model()
    clf.display_dataset_information()
    clf.train_model()
    clf.evaluate_model()
    clf.export_model()
