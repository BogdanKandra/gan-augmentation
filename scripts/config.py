import os


LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')

PROJECT_PATH = os.getcwd()
while os.path.basename(PROJECT_PATH) != 'gan-augmentation':
    PROJECT_PATH = os.path.dirname(PROJECT_PATH)
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')
CNN_RESULTS_PATH = os.path.join(RESULTS_PATH, 'cnn-training')
GAN_RESULTS_PATH = os.path.join(RESULTS_PATH, 'gan-training')

TRAIN_SET_PERCENTAGE = 0.85
VALID_SET_PERCENTAGE = 0.15
RANDOM_STATE = 29
NUM_EPOCHS_WEAK = 10
NUM_EPOCHS_STRONG = 50
BATCH_SIZE_WEAK = 32
BATCH_SIZE_STRONG = 16
