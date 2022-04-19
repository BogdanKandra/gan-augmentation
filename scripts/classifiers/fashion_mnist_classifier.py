import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print('Train Set Shape: X_train.shape={}, y_train.shape={}'.format(X_train.shape, y_train.shape))
print('Test Set Shape: X_test.shape={}, y_test.shape={}'.format(X_test.shape, y_test.shape))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

plt.show()
