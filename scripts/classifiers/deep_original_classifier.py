import json
import os
from scripts.classifiers.fashion_mnist_classifier import FashionMNISTClassifier
from scripts import config, utils
import numpy as np
import sklearn.metrics as sk_metrics
from tensorflow.python.keras.activations import relu, softmax
from tensorflow.python.keras.layers import Dense, Flatten, InputLayer
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizers import serialize
from tensorflow.python.keras.utils.np_utils import to_categorical


LOGGER = utils.get_logger(__name__)


class DNNOriginalClassifier(FashionMNISTClassifier):
    """ Class representing a good classifier for the original Fashion-MNIST dataset, using a deep neural network """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and encoding the labels """
        self.X_train = self.X_train.reshape((*self.X_train.shape, 1)).astype(float) / 255.0
        self.X_valid = self.X_valid.reshape((*self.X_valid.shape, 1)).astype(float) / 255.0
        self.X_test = self.X_test.reshape((*self.X_test.shape, 1)).astype(float) / 255.0

        self.y_train = to_categorical(self.y_train)
        self.y_valid = to_categorical(self.y_valid)
        self.y_test = to_categorical(self.y_test)

    def build_model(self) -> None:
        """ Defines the classifier model structure and stores it as an instance attribute. The model used here is a deep
         neural network, consisting of the Input and Output layers and 3 hidden layers in between, with a vanilla SGD as
         optimizer """
        self.model = Sequential(name='DNNOriginalClassifier')
        self.model.add(InputLayer(input_shape=(self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]),
                                  dtype=float,
                                  name='original_image'))
        self.model.add(Flatten())
        self.model.add(Dense(units=256, activation=relu, kernel_initializer='he_uniform'))
        self.model.add(Dense(units=64, activation=relu, kernel_initializer='he_uniform'))
        self.model.add(Dense(units=16, activation=relu, kernel_initializer='he_uniform'))
        self.model.add(Dense(units=10, activation=softmax, kernel_initializer='he_uniform'))
        optimizer = SGD(learning_rate=0.05)
        self.model.compile(optimizer=optimizer,
                           loss=CategoricalCrossentropy(),
                           metrics=[CategoricalAccuracy(), Precision(), Recall()])

    def train_model(self) -> None:
        """ Performs the training and evaluation of this classifier, on both the train set and the validation set.
         The loss function to be optimised is the Categorical Cross-entropy loss and the measured metric is Accuracy,
          which is appropriate for our problem, because the dataset classes are balanced.  """
        self.__training_history = self.model.fit(x=self.X_train, y=self.y_train, batch_size=config.BATCH_SIZE_DEEP,
                                                 epochs=config.NUM_EPOCHS_DEEP, verbose=1,
                                                 validation_data=(self.X_valid, self.y_valid)).history
        self.__test_accuracy = self.model.evaluate(x=self.X_test, y=self.y_test, batch_size=config.BATCH_SIZE_DEEP,
                                                   verbose=1, return_dict=True)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by plotting training and validation accuracy and loss and generating
        the classification report and confusion matrix """
        results_subdirectory_name = self.create_current_run_directory('result')
        utils.plot_results(results_subdirectory_name, self.__training_history)
        results_name = 'Training Results.txt'
        with open(os.path.join(config.CLASSIFIER_RESULTS_PATH, results_subdirectory_name, results_name), 'w') as f:
            f.write(json.dumps(self.__training_history, indent=4))

        # Save the test results
        results_name = 'Test Results.txt'
        with open(os.path.join(config.CLASSIFIER_RESULTS_PATH, results_subdirectory_name, results_name), 'w') as f:
            f.write(json.dumps(self.__test_accuracy, indent=4))

        # Generate the classification report
        predictions = self.model.predict(x=self.X_test, batch_size=config.BATCH_SIZE_DEEP, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_pred_categorical = to_categorical(y_pred)
        class_labels = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        report = sk_metrics.classification_report(self.y_test, y_pred_categorical, target_names=class_labels)
        report_name = 'Classification Report.txt'
        with open(os.path.join(config.CLASSIFIER_RESULTS_PATH, results_subdirectory_name, report_name), 'w') as f:
            f.write(report)

        # Generate the confusion matrix
        self.y_test = np.argmax(self.y_test, axis=1)
        cm = sk_metrics.confusion_matrix(self.y_test, y_pred, labels=list(range(10)))
        utils.plot_confusion_matrix(cm, results_subdirectory_name, class_labels)

        # Generate a file containing model information and parameters
        training_info_name = 'Training Information.txt'
        training_info = {
            'batch_size': config.BATCH_SIZE_DEEP,
            'num_epochs': config.NUM_EPOCHS_DEEP,
            'model': json.loads(self.model.to_json()),
            'optimizer': str(serialize(self.model.optimizer))
        }
        with open(os.path.join(config.CLASSIFIER_RESULTS_PATH, results_subdirectory_name, training_info_name), 'w') as f:
            f.write(json.dumps(training_info, indent=4))


if __name__ == '__main__':
    clf = DNNOriginalClassifier()
    clf.preprocess_dataset()
    clf.build_model()
    clf.display_model()
    clf.display_dataset_information()
    # clf.train_model()
    # clf.evaluate_model()
    # clf.export_model()
