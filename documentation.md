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

<br>

## Models
The <b><i>models</i></b> directory contains the trained classifiers and generators, saved in Tensorflow.js format. This format is needed to deploy models on the Web. The trained models are saved automatically after each training run is completed and evaluated.

## Notebooks
The <b><i>notebooks</i></b> directory contains Jupyter notebooks which perform various tasks related to the project, such as training models and visualizing results.

## Results
The <b><i>results</i></b> directory contains training results for the classifiers and generators; these are generated automatically for each training run and are saved in a separate directory according to the model architecture and run index. They include:
- <i>classification report</i> - text file containing the precision, recall, f1 score and number of samples per each of the 10 classes for the current training run
- <i>confusion matrix</i> - image file containing the confusion matrix for the current training run
- train / test / valid results - loss, accuracy, precision, recall of the current training run
- train and validation accuracy and loss curves (image)
- model information - text file containing the batch size, number of epochs, model architecture and optimizer hyperparameters

The naming scheme for results directories is <b><i>\<Model Architecture\>\<Dataset Type\>\<Model Type\> Run \<Index\></i></b>; the part before <b><i>Run</i></b> is set by the name of the underlying class handling model creation and training. The <b><i>\<Index\></i></b> part is computed automatically based on how many directories belonging to the same type of models are present.

## Scripts
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
        - <b>\<Model Architecture\>\<Dataset Type\>\<Model Type\></b> concrete classes, differing by the architecture of the model (Shallow NN / Deep NN / CNN / Transfer Learning NN), the type of dataset used (original / augmented) and by the type of the model (Classifier / Generator)

## Tests
The tests directory contains unit tests for the scripts.

<br>

# References
- https://www.quora.com/Do-convolutional-neural-networks-learn-to-be-spatially-invariant-at-the-last-layer-of-the-network-fully-connected-layer-Convolution-layers-produce-spatially-equivariant-output-but-what-about-the-spatial-invariance
- https://towardsdatascience.com/translational-invariance-vs-translational-equivariance-f9fbc8fca63a
- https://pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/  (Training CNN on Fashion-MNIST)
- https://medium.com/@mjbhobe/classifying-fashion-with-a-keras-cnn-achieving-94-accuracy-part-2-a5bd7a4e7e5a  (Training CNN on Fashion-MNIST)
- https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights  (Transfer Learning)