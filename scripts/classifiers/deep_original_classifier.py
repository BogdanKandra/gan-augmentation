from scripts.classifiers import FashionMNISTClassifier
from scripts import config, utils
from tensorflow.python.keras.activations import relu, softmax
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense, Flatten, InputLayer
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
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
        optimizer = SGD(learning_rate=0.01)
        self.model.compile(optimizer=optimizer,
                           loss=CategoricalCrossentropy(),
                           metrics=[CategoricalAccuracy(), Precision(), Recall()])

    def train_model(self) -> None:
        """ Performs the training and evaluation of this classifier, on both the train set and the validation set.
         The loss function to be optimised is the Categorical Cross-entropy loss and the measured metric is Accuracy,
          which is appropriate for our problem, because the dataset classes are balanced.  """
        self._create_current_run_directory()
        es_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

        self.__training_history = self.model.fit(x=self.X_train, y=self.y_train, batch_size=config.BATCH_SIZE_DEEP,
                                                 epochs=config.NUM_EPOCHS_DEEP, verbose=1, callbacks=[es_callback],
                                                 validation_data=(self.X_valid, self.y_valid)).history
        self.__test_accuracy = self.model.evaluate(x=self.X_test, y=self.y_test, batch_size=config.BATCH_SIZE_DEEP,
                                                   verbose=1, return_dict=True)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by plotting training and validation accuracy and loss and generating
        the classification report and confusion matrix """
        super().evaluate_model(config.DEEP_CLF_HYPERPARAMS)


if __name__ == '__main__':
    clf = DNNOriginalClassifier()
    clf.preprocess_dataset()
    clf.build_model()
    clf.display_model()
    clf.display_dataset_information()
    # clf.train_model()
    # clf.evaluate_model()
    # clf.export_model()
