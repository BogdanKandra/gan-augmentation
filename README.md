# gan-augmentation
Studying the utility of GANs in augmenting datasets used for classification problems

<br><br><br>

# Project Structure

- models
    - classifiers
    - generators
- notebooks
- results
    - classifiers
    - generators
- scripts
    - classifiers
    - generators
    - interfaces
- tests

<br><br><br>

# Models

The <b><i>models</i></b> directory contains the trained classifiers and generators, saved in Tensorflow.js format. This format is needed to deploy models on the Web. The trained models are saved automatically after each training run is completed and evaluated. <br><br><br>

# Notebooks

The <b><i>notebooks</i></b> directory contains Jupyter notebooks which perform various tasks related to the project, such as training models and visualizing results. <br><br><br>

# Results

The <b><i>results</i></b> directory contains training results for the classifiers and generators; these are generated automatically for each training run and are saved in a separate directory according to the model architecture and run index. They  include:
- <i>classification report</i> - text file containing the precision, recall, f1 score and number of samples per each of the 10 classes for the current training run
- <i>confusion matrix</i> - image file containing the confusion matrix for the current training run
- train / test / valid results - loss, accuracy, precision, recall of the current training run
- train and validation accuracy and loss curves (image)
- model information - text file containing the batch size, number of epochs, model architecture and optimizer hyperparameters

The naming scheme for results directories is <b><i>\<Model Architecture\>\<Dataset Type\>\<Model Type\> Run \<Index\></i></b>; the part before <b><i>Run</i></b> is set by the name of the underlying class handling model creation and training. The <b><i>\<Index\></i></b> part is computed automatically based on how many directories belonging to the same type of models are present. <br><br><br>

# Scripts

The scripts directory contains:
- <i>classifiers</i> directory - scripts defining classifier models
- <i>generators</i> directory - scripts defining generator models
- <i>interfaces</i> directory - script containing the <b>FashionMNISTModel</b> interface
- <i>config.py</i> - script containing project configuration constants, such as project paths and model training hyperparameter default values
- <i>utils.py</i> - script containing utilitary functions

Architecture:
- <b>FashionMNISTModel</b> interface
    - <b>FashionMNISTClassifier</b> abstract class for classifiers
    - <b>FashionMNISTGenerator</b> abstract class for generators
        - <b>\<Model Architecture\>\<Dataset Type\>\<Model Type\></b> concrete classes, differing by the architecture of the model (Shallow NN / Deep NN / CNN / Transfer Learning NN), the type of dataset used (original / augmented) and by the type of the model (Classifier / Generator) <br><br><br>

# Tests

The tests directory contains unit tests for the scripts. <br><br><br>

# TODO

- Write project installation instructions (see other repos of mine) [???]
    - conda create --name <env_name> python=3.11
    - conda activate <env_name>
    - pip install -r requirements.txt (from the project dir)
- Switch everything from TensorFlow to PyTorch
- Remove magic numbers
- Write tests for the classifier scripts ???
- Maybe create a class structure for representing the training data ?
- Implement generator models
- Augment dataset using generator models
- Train classifiers on augmented datasets
- Compare results