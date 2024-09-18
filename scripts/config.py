import os
from enum import Enum
from pathlib import Path


# Enums
class ClassifierDataset(Enum):
    FASHION_MNIST = 1
    FASHION_MNIST_GAN = 2
    FASHION_MNIST_DCGAN = 3
    FASHION_MNIST_WGAN_GP = 4
    FASHION_MNIST_DDPM = 5
    FASHION_MNIST_DDIM = 6
    CIFAR_10 = 7
    CIFAR_10_GAN = 8
    CIFAR_10_DCGAN = 9
    CIFAR_10_WGAN_GP = 10
    CIFAR_10_DDPM = 11
    CIFAR_10_DDIM = 12


class GeneratorDataset(Enum):
    FASHION_MNIST = 1
    CIFAR_10 = 2


class ClassifierType(Enum):
    SHALLOW = 1
    DEEP = 2
    CONVOLUTIONAL = 3
    TRANSFER_LEARNING = 4


class GeneratorType(Enum):
    GAN = 1
    DCGAN = 2
    WGAN_GP = 3
    DDPM = 4
    DDIM = 5


LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")


# Paths
PROJECT_PATH = Path.cwd()
while PROJECT_PATH.stem != "gan-augmentation":
    PROJECT_PATH = PROJECT_PATH.parent
ARTIFACTS_PATH = PROJECT_PATH / "artifacts"
CLASSIFIERS_PATH = ARTIFACTS_PATH / "classifiers"
GENERATORS_PATH = ARTIFACTS_PATH / "generators"
RESULTS_PATH = PROJECT_PATH / "results"
CLASSIFIER_RESULTS_PATH = RESULTS_PATH / "classifiers"
GENERATOR_RESULTS_PATH = RESULTS_PATH / "generators"


# Datasets labels and sizes
FASHION_MNIST_CLASS_LABELS = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat",
                              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
CIFAR_10_CLASS_LABELS = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                         "Dog", "Frog", "Horse", "Ship", "Truck"]
FASHION_MNIST_SHAPE = (1, 28, 28)
CIFAR_10_SHAPE = (3, 32, 32)


# Training hyperparameters
TRAIN_SET_PERCENTAGE = 0.85
VALID_SET_PERCENTAGE = 0.15
RANDOM_STATE = 29

SHALLOW_CLF_HYPERPARAMS = {
    "BATCH_SIZE": 16,
    "EARLY_STOPPING_TOLERANCE": 5,
    "LEARNING_RATE": 0.01,
    "NUM_EPOCHS": 20
}

DEEP_CLF_HYPERPARAMS = {
    "BATCH_SIZE": 32,
    "EARLY_STOPPING_TOLERANCE": 5,
    "LEARNING_RATE": 0.01,
    "NUM_EPOCHS": 30
}

CONVOLUTIONAL_CLF_HYPERPARAMS = {
    "BATCH_SIZE": 32,
    "EARLY_STOPPING_TOLERANCE": 5,
    "LEARNING_RATE": 0.0001,
    "NUM_EPOCHS": 30
}

EFFICIENTNET_CLF_HYPERPARAMS = {
    "BATCH_SIZE": 16,
    "EARLY_STOPPING_TOLERANCE": 10,
    "LEARNING_RATE": 0.001,
    "NUM_EPOCHS": 50
}

GAN_GEN_HYPERPARAMS = {
    "BATCH_SIZE": 128,
    "EARLY_STOPPING_TOLERANCE": 20,
    "LEARNING_RATE": 0.00001,
    "NUM_EPOCHS": 200
}

DCGAN_GEN_HYPERPARAMS = {
    "BATCH_SIZE": None,
    "EARLY_STOPPING_TOLERANCE": None,
    "LEARNING_RATE": None,
    "NUM_EPOCHS": None
}

WGAN_GP_GEN_HYPERPARAMS = {
    "BATCH_SIZE": None,
    "EARLY_STOPPING_TOLERANCE": None,
    "LEARNING_RATE": None,
    "NUM_EPOCHS": None
}

DDPM_GEN_HYPERPARAMS = {
    "BATCH_SIZE": None,
    "EARLY_STOPPING_TOLERANCE": None,
    "LEARNING_RATE": None,
    "NUM_EPOCHS": None
}

DDIM_GEN_HYPERPARAMS = {
    "BATCH_SIZE": None,
    "EARLY_STOPPING_TOLERANCE": None,
    "LEARNING_RATE": None,
    "NUM_EPOCHS": None
}

L2_LOSS_LAMBDA_2 = 0.0002
