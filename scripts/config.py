import os
from enum import Enum
from pathlib import Path


class ClassifierType(Enum):
    SHALLOW = 1
    DEEP = 2
    CONVOLUTIONAL = 3
    TRANSFER_LEARNING = 4


LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')

PROJECT_PATH = Path.cwd()
while PROJECT_PATH.stem != 'gan-augmentation':
    PROJECT_PATH = PROJECT_PATH.parent
MODELS_PATH = PROJECT_PATH / 'models'
CLASSIFIERS_PATH = MODELS_PATH / 'classifiers'
GENERATORS_PATH = MODELS_PATH / 'generators'
RESULTS_PATH = PROJECT_PATH / 'results'
CLASSIFIER_RESULTS_PATH = RESULTS_PATH / 'classifiers'
GENERATOR_RESULTS_PATH = RESULTS_PATH / 'generators'

CLASS_LABELS = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

TRAIN_SET_PERCENTAGE = 0.85
VALID_SET_PERCENTAGE = 0.15
RANDOM_STATE = 29

SHALLOW_CLF_HYPERPARAMS = {
    'NUM_EPOCHS': 30,
    'BATCH_SIZE': 16
}

DEEP_CLF_HYPERPARAMS = {
    'NUM_EPOCHS': 30,
    'BATCH_SIZE': 32
}

CONVOLUTIONAL_CLF_HYPERPARAMS = {
    'NUM_EPOCHS': 30,
    'BATCH_SIZE': 32
}

EFFICIENTNET_CLF_HYPERPARAMS = {
    'NUM_EPOCHS': 50,
    'BATCH_SIZE': 32
}

L2_LOSS_LAMBDA_2 = 0.0002

EFFICIENT_NET_HEIGHT = 224
EFFICIENT_NET_WIDTH = 224
