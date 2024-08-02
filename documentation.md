# TODO
- Replace TensorFlow with PyTorch
    - Rewrite efficientnet_original_classifier

- New branch: Refactoring
	- Update docs throughout the project
	- scripts/classifiers:
		- Rename the 4 x Classifier concrete classes
			- Since these do not depend on the dataset, the names are:
				- shallow_classifier.py
				- deep_classifier.py
				- convolutional_classifier.py
				- efficientnet_classifier.py
		- Rename the 1 x Classifier abstract class
			- Since this represents classifiers on TorchVision datasets, its name is:
				- torchvision_dataset_classifier.py

	- scripts/generators:
		- Create the 4 x Generator concrete classes
			- Since these do not depend on the dataset, the names are:
				- gan_generator.py
				- dcgan_generator.py
				- ddpm_generator.py
				- ddim_generator.py
		- Create 1 x Generator abstract class
			- Since this represents generators on TorchVision datasets, its name is:
				- torchvision_dataset_generator.py
			- The constructor takes in a str representing the TorchVision dataset name

	- scripts/interfaces:
		- Rename the 1 x TorchVisionDatasetModel interface
			- Since this represents models trained on TorchVision datasets, its name
			is:
				- torchvision_dataset_model.py

	- _create_current_run_directory()
		- When saving classifier models and training results, each directory is named
			according to the user selected classifier type and dataset
		- When saving generator models and training results, each directory is named
			according to the user selected generator type and dataset

	- display_dataset_information()
		- Also plot the label for each sample



- Add stuff learned from Krish Naik video:
	- Integrate MlFlow
	- Remove magic numbers by adding config file for constants and hyperparams
	- Update README file with details on how to train and perform inference and stuff
	- Watch video again

- Prepare training on GPU:
	- Write notebook for testing GPU availability
	- Update code so that it is device-aware
	- Update training notebooks so that they also use the GPU if available

- Implement GANs:
	- VanillaGAN
	- DCGAN
	- Implement both GANs as conditional GANs
	- Implement both loss functions: BCE loss / W-loss
	- Use the Frechet Inception Distance (FID) and Inception Score (IS) eval metrics

- Implement Diffusion Models:
	- DDPM
	- DDIM
	- Implement both DMs as conditional DMs
	- Loss functions?
	- Use the Frechet Inception Distance (FID) and Inception Score (IS) eval metrics

- Augment datasets using generator models
	- How exactly ??

- Train the 4 classifiers on augmented datasets

- Analyze results

- Write Jupyter Notebooks:
	- 1 x Notebook for training and evaluating generator models
		- User chooses generator type (VanillaGAN / DCGAN / DDPM / DDIM)
		- User chooses dataset (FashionMNIST / CIFAR-10)
	- 1 x Notebook for running inference on the best generator model
	- 1 x Notebook for running inference on the best classifier model
	- 1 x Notebook for testing GPU availability



- Integrate TensorBoard ???
- Add L2 regularization to the CNN classifier?    # l2 = regularizers.l2(config.L2_LOSS_LAMBDA_2)
- Maybe create a class structure for representing the training data ?
- Add inference function with ONNX Runtime (Check third ONNX reference in documentation)
- Add quantization function
- Write tests for the classifier scripts ???
- Deployment of model - https://pytorch.org/serve/

<br>

# Project Structure
- artifacts
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
    - models
- tests

<br>

## Artifacts
The <b><i>artifacts</i></b> directory contains the trained classifiers and generators, saved in ONNX format. The trained models are saved automatically after each training run is completed and evaluated.

## Notebooks
The <b><i>notebooks</i></b> directory contains Jupyter notebooks which perform various tasks related to the project, such as training models and visualizing results.

## Results
The <b><i>results</i></b> directory contains training results for the classifiers and generators; these are generated automatically for each training run and are saved in a separate directory according to the model architecture and run index. They include:
- <i>classification report</i> - text file containing the precision, recall, f1 score and number of samples per each of the 10 classes for the current training run
- <i>confusion matrix</i> - image file containing the confusion matrix for the current training run
- train / test / valid results - loss, accuracy, precision, recall of the current training run
- train and validation accuracy and loss curves (image)
- model information - text file containing the model architecture, loss and optimizer functions, and chosen hyperparameters

The naming scheme for results directories is <b><i>\<Model Architecture\>\<Dataset Type\>\<Model Type\> Run \<Index\></i></b>; the part before <b><i>Run</i></b> is set by the name of the underlying class handling model creation and training. The <b><i>\<Index\></i></b> part is computed automatically based on how many directories belonging to the same type of models are present.

## Scripts
The scripts directory contains:
- <i>classifiers</i> directory - scripts defining classifier models
    - shallow_classifier, deep_classifier, convolutional_classifier, efficientnet_classifier
- <i>generators</i> directory - scripts defining generator models
    - vanilla_gan, deep_convolutional_gan, ddpm_diffusion, ddim_diffusion
- <i>interfaces</i> directory - script containing the <b>FashionMNISTModel</b> interface
- <i>models</i> directory - scripts defining classes for each classifier and generator network
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

# Datasets
- There are two main datasets used in this project - Fashion-MNIST and CIFAR-10
- The generators are trained on both datasets
- The classifiers are trained on both the original datasets and versions augmented using each of the trained generator

- Dataset Loading and Splitting:
    - Done in the abstract class constructors, using the dataset parameter
    - The dataset_name has one of the following values:
    ['fashion_mnist', 'fashion_mnist_gan', 'fashion_mnist_dcgan',
        'fashion_mnist_ddpm', 'fashion_mnist_ddim', 'cifar-10', 'cifar-10_gan',
        'cifar-10_dcgan', 'cifar-10_ddpm', 'cifar-10_ddim']
    - If dataset_name is 'fashion_mnist' or 'cifar-10', the original datasets
        are loaded; otherwise, the original datasets are augmented using
        the specified generator model.

<br>

# References
- https://www.quora.com/Do-convolutional-neural-networks-learn-to-be-spatially-invariant-at-the-last-layer-of-the-network-fully-connected-layer-Convolution-layers-produce-spatially-equivariant-output-but-what-about-the-spatial-invariance
- https://towardsdatascience.com/translational-invariance-vs-translational-equivariance-f9fbc8fca63a
- https://pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/  (Training CNN on Fashion-MNIST)
- https://medium.com/@mjbhobe/classifying-fashion-with-a-keras-cnn-achieving-94-accuracy-part-2-a5bd7a4e7e5a  (Training CNN on Fashion-MNIST)
- https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights  (Transfer Learning)

- https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html (ONNX export)
- https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model (ONNX export)
- https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html (ONNX export)
