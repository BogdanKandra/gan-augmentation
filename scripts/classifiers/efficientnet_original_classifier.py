import json
import os
from scripts.classifiers.fashion_mnist_classifier import FashionMNISTClassifier
from scripts import config, utils
import numpy as np
import sklearn.metrics as sk_metrics
from tensorflow.python.keras.activations import softmax
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Dense, Dropout, InputLayer, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Resizing
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizers import serialize
from tensorflow.python.keras.utils.np_utils import to_categorical


LOGGER = utils.get_logger(__name__)


class EfficientNetOriginalClassifier(FashionMNISTClassifier):
    """ Class representing a <TBA> classifier for the original Fashion-MNIST dataset, using the transfer learning
     approach """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and encoding the labels """
        # Preprocess input
        self.X_train = np.expand_dims(self.X_train, axis=3).astype(float)
        self.X_valid = np.expand_dims(self.X_valid, axis=3).astype(float)
        self.X_test = np.expand_dims(self.X_test, axis=3).astype(float)

        # The input images expected by the EfficientNet models must have 3 channels, so we convert the data
        self.X_train = np.concatenate([self.X_train] * 3, axis=3)
        self.X_valid = np.concatenate([self.X_valid] * 3, axis=3)
        self.X_test = np.concatenate([self.X_test] * 3, axis=3)

        self.y_train = to_categorical(self.y_train)
        self.y_valid = to_categorical(self.y_valid)
        self.y_test = to_categorical(self.y_test)

    def build_model(self) -> None:
        """ Defines the classifier model structure and stores it as an instance attribute. The model used here is the
        headless EfficientNetB3 pretrained network with a new classifier head consisting of Global Average Pooling,
         Dropout and Softmax. Early stopping and TensorBoard callbacks are also implemented """
        feature_extractor = EfficientNetB0(include_top=False, weights='imagenet')
        feature_extractor.trainable = False

        self.model = Sequential(name='EfficientNetOriginalClassifier')
        self.model.add(InputLayer(input_shape=(self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]),
                                  dtype=float,
                                  name='original_image'))
        self.model.add(Resizing(config.EFFICIENT_NET_HEIGHT, config.EFFICIENT_NET_WIDTH))

        self.model.add(feature_extractor)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.2))

        self.model.add(Dense(10, activation=softmax, kernel_initializer='he_uniform'))

        optimizer = Adam(learning_rate=0.001, decay=0.01 / config.NUM_EPOCHS_EFFICIENTNET)
        self.model.compile(optimizer=optimizer,
                           loss=CategoricalCrossentropy(),
                           metrics=[CategoricalAccuracy(), Precision(), Recall()])

    def train_model(self) -> None:
        """ Performs the training and evaluation of this classifier, on both the train set and the validation set.
         The loss function to be optimised is the Categorical Cross-entropy loss and the measured metric is Accuracy,
          which is appropriate for our problem, because the dataset classes are balanced.  """
        self.create_current_run_directory()
        logs_path = os.path.join(config.CLASSIFIER_RESULTS_PATH, self.results_subdirectory, 'logs')
        es_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        tb_callback = TensorBoard(log_dir=logs_path)

        self.__training_history = self.model.fit(x=self.X_train, y=self.y_train,
                                                 batch_size=config.BATCH_SIZE_EFFICIENTNET,
                                                 epochs=config.NUM_EPOCHS_EFFICIENTNET, verbose=1,
                                                 callbacks=[es_callback, tb_callback],
                                                 validation_data=(self.X_valid, self.y_valid)).history
        self.__test_accuracy = self.model.evaluate(x=self.X_test, y=self.y_test,
                                                   batch_size=config.BATCH_SIZE_EFFICIENTNET,
                                                   verbose=1, return_dict=True)

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
        predictions = self.model.predict(x=self.X_test, batch_size=config.BATCH_SIZE_EFFICIENTNET, verbose=1)
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
        training_info_path = os.path.join(config.CLASSIFIER_RESULTS_PATH, self.results_subdirectory, training_info_name)
        try:
            model_str = json.loads(self.model.to_json())
        except NotImplementedError:
            model_str = ''
        training_info = {
            'batch_size': config.BATCH_SIZE_EFFICIENTNET,
            'num_epochs': config.NUM_EPOCHS_EFFICIENTNET,
            'model': model_str,
            'optimizer': str(serialize(self.model.optimizer))
        }
        with open(training_info_path, 'w') as f:
            f.write(json.dumps(training_info, indent=4))


if __name__ == '__main__':
    clf = EfficientNetOriginalClassifier()
    clf.preprocess_dataset()
    clf.display_dataset_information()
    clf.build_model()
    clf.display_model()
    clf.train_model()
    clf.evaluate_model()
    clf.export_model()
