import json
import os
from scripts.classifiers.fashion_mnist_classifier import FashionMNISTClassifier
from scripts import config, utils
import numpy as np
import sklearn.metrics as sk_metrics
from tensorflow.python.keras.activations import relu, softmax
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPooling2D
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizers import serialize
import tensorflow.python.keras.regularizers as regularizers
from tensorflow.python.keras.utils.np_utils import to_categorical


LOGGER = utils.get_logger(__name__)


class CNNOriginalClassifier(FashionMNISTClassifier):
    """ Class representing a strong classifier for the original Fashion-MNIST dataset, using a convolutional neural
     network """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and encoding the labels """
        self.X_train = self.X_train.reshape((*self.X_train.shape, 1)).astype(float) / 255.0
        self.X_valid = self.X_valid.reshape((*self.X_valid.shape, 1)).astype(float) / 255.0
        self.X_test = self.X_test.reshape((*self.X_test.shape, 1)).astype(float) / 255.0

        self.y_train = to_categorical(self.y_train)
        self.y_valid = to_categorical(self.y_valid)
        self.y_test = to_categorical(self.y_test)

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
        self.model.add(Conv2D(filters=64, kernel_size=3, activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(rate=0.2))

        self.model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(Conv2D(filters=128, kernel_size=3, activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(rate=0.3))

        self.model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(Conv2D(filters=256, kernel_size=3, activation=relu,
                              kernel_initializer='he_uniform', kernel_regularizer=l2))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Dropout(rate=0.4))

        self.model.add(Flatten())

        self.model.add(Dense(units=1024, activation=relu, kernel_initializer='he_uniform'))
        self.model.add(Dropout(rate=0.3))
        self.model.add(Dense(units=512, activation=relu, kernel_initializer='he_uniform'))
        self.model.add(Dropout(rate=0.2))

        self.model.add(Dense(10, activation=softmax, kernel_initializer='he_uniform'))

        optimizer = Adam(learning_rate=0.0001, decay=0.0001 / config.NUM_EPOCHS_CONVOLUTIONAL)
        self.model.compile(optimizer=optimizer,
                           loss=CategoricalCrossentropy(),
                           metrics=[CategoricalAccuracy(), Precision(), Recall()])

    def train_model(self) -> None:
        """ Performs the training and evaluation of this classifier, on both the train set and the validation set.
         The loss function to be optimised is the Categorical Cross-entropy loss and the measured metric is Accuracy,
          which is appropriate for our problem, because the dataset classes are balanced.  """
        self.create_current_run_directory()
        logs_path = os.path.join(config.CLASSIFIER_RESULTS_PATH, self.results_subdirectory, 'logs')
        es_callback = EarlyStopping(monitor='val_loss', patience=5)
        tb_callback = TensorBoard(log_dir=logs_path)

        self.__training_history = self.model.fit(x=self.X_train, y=self.y_train,
                                                 batch_size=config.BATCH_SIZE_CONVOLUTIONAL,
                                                 epochs=config.NUM_EPOCHS_CONVOLUTIONAL, verbose=1,
                                                 callbacks=[es_callback, tb_callback],
                                                 validation_data=(self.X_valid, self.y_valid)).history
        self.__test_accuracy = self.model.evaluate(x=self.X_test, y=self.y_test,
                                                   batch_size=config.BATCH_SIZE_CONVOLUTIONAL,
                                                   verbose=1, callbacks=[es_callback, tb_callback], return_dict=True)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by plotting training and validation accuracy and loss and generating
        the classification report and confusion matrix """
        utils.plot_results(self.results_subdirectory, self.__training_history)
        results_name = 'Training Results.txt'
        with open(os.path.join(config.CLASSIFIER_RESULTS_PATH, self.results_subdirectory, results_name), 'w') as f:
            f.write(json.dumps(self.__training_history, indent=4))

        # Save the test results
        results_name = 'Test Results.txt'
        with open(os.path.join(config.CLASSIFIER_RESULTS_PATH, self.results_subdirectory, results_name), 'w') as f:
            f.write(json.dumps(self.__test_accuracy, indent=4))

        # Generate the classification report
        logs_path = os.path.join(config.CLASSIFIER_RESULTS_PATH, self.results_subdirectory, 'logs')
        es_callback = EarlyStopping(monitor='val_loss', patience=5)
        tb_callback = TensorBoard(log_dir=logs_path)
        predictions = self.model.predict(x=self.X_test, batch_size=config.BATCH_SIZE_CONVOLUTIONAL, verbose=1,
                                         callbacks=[es_callback, tb_callback])
        y_pred = np.argmax(predictions, axis=1)
        y_pred_categorical = to_categorical(y_pred)
        class_labels = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        report = sk_metrics.classification_report(self.y_test, y_pred_categorical, target_names=class_labels)
        report_name = 'Classification Report.txt'
        with open(os.path.join(config.CLASSIFIER_RESULTS_PATH, self.results_subdirectory, report_name), 'w') as f:
            f.write(report)

        # Generate the confusion matrix
        self.y_test = np.argmax(self.y_test, axis=1)
        cm = sk_metrics.confusion_matrix(self.y_test, y_pred, labels=list(range(10)))
        utils.plot_confusion_matrix(cm, self.results_subdirectory, class_labels)

        # Generate a file containing model information and parameters
        training_info_name = 'Training Information.txt'
        try:
            model_str = json.loads(self.model.to_json())
        except NotImplementedError:
            model_str = ''
        training_info = {
            'batch_size': config.BATCH_SIZE_CONVOLUTIONAL,
            'num_epochs': config.NUM_EPOCHS_CONVOLUTIONAL,
            'model': model_str,
            'optimizer': str(serialize(self.model.optimizer))
        }
        training_info_path = os.path.join(config.CLASSIFIER_RESULTS_PATH, self.results_subdirectory, training_info_name)
        with open(training_info_path, 'w') as f:
            f.write(json.dumps(training_info, indent=4))


if __name__ == '__main__':
    clf = CNNOriginalClassifier()
    clf.preprocess_dataset()
    clf.build_model()
    clf.display_model()
    clf.display_dataset_information()
    clf.train_model()
    clf.evaluate_model()
    clf.export_model()
