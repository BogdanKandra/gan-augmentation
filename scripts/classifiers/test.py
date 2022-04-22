import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


def show_dataset_information():
    print('Train Set Shape: X_train.shape={}, y_train.shape={}'.format(X_train.shape, y_train.shape))
    print('Test Set Shape: X_test.shape={}, y_test.shape={}'.format(X_test.shape, y_test.shape))


def show_dataset_sample():
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.show()


X_train = X_train.reshape((*X_train.shape, 1))
X_test = X_test.reshape((*X_test.shape, 1))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train_norm = X_train.astype(np.float)
X_test_norm = X_test_astype(np.float)
X_train_norm = X_train_norm / 255.0
X_test_norm = X_test_norm / 255.0
