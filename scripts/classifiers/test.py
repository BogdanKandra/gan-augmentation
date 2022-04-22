from scripts.classifiers.fashion_mnist_classifier import FashionMNISTClassifier
import numpy as np
from tensorflow.keras.utils import to_categorical


class TestClassifier(FashionMNISTClassifier):
    """ Test class representing a concrete implementation of the base classifier """
    def __init__(self):
        """ This just calls the base class' constructor """
        super().__init__()

    def preprocess_dataset(self) -> None:
        """ This reshapes the dataset and encodes the labels """
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        x_temp_train = self.X_train.reshape((*self.X_train.shape, 1)).astype(float)
        x_temp_test = self.X_test.reshape((*self.X_test.shape, 1)).astype(float)
        self.X_train = x_temp_train / 255.0
        self.X_test = x_temp_test / 255.0


clf = TestClassifier()
clf.display_dataset_information()
clf.display_dataset_sample(3)
clf.display_model()
clf.preprocess_dataset()
clf.display_dataset_information()
