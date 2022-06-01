import os


LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')

PROJECT_PATH = os.getcwd()
while os.path.basename(PROJECT_PATH) != 'gan-augmentation':
    PROJECT_PATH = os.path.dirname(PROJECT_PATH)
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')
CLASSIFIERS_PATH = os.path.join(MODELS_PATH, 'classifiers')
GENERATORS_PATH = os.path.join(MODELS_PATH, 'generators')
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')
CLASSIFIER_RESULTS_PATH = os.path.join(RESULTS_PATH, 'classifiers')
GENERATOR_RESULTS_PATH = os.path.join(RESULTS_PATH, 'generators')

TRAIN_SET_PERCENTAGE = 0.85
VALID_SET_PERCENTAGE = 0.15
RANDOM_STATE = 29
NUM_EPOCHS_SHALLOW = 1
NUM_EPOCHS_DEEP = 10
NUM_EPOCHS_CONVOLUTIONAL = 30
BATCH_SIZE_SHALLOW = 16
BATCH_SIZE_DEEP = 32
BATCH_SIZE_CONVOLUTIONAL = 64
L2_LOSS_LAMBDA_2 = 0.0002
