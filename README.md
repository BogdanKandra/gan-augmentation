# gan-augmentation
Studying the utility of GANs in augmenting datasets used for classification problems

<br>

## Steps for running
- Create a virtual environment for the project: `conda create -n <env_name> python=3.10`
- Activate the virtual environment: `conda activate <env_name>`
- Install the development dependencies: `pip install -r requirements.txt`
- Run any of the provided Jupyter notebooks, within the created environment

<br>

## Steps for testing
- Create a virtual environment for the project (or activate the one created for the step above)
- Install the testing dependencies: `pip install -r test-requirements.txt`
- Run the [pytest](https://docs.pytest.org/en/8.1.x/) test suit from the project base directory: `python -m pytest tests`

<br>

## TODO
- Replace TensorFlow with PyTorch
- Update the call to torchinfo.summary()

- Add export to ONNX
- Add stuff learned from Krish Naik video:
    - Integrate MlFlow
    - Remove magic numbers by adding config file for constants and hyperparams
- Update README file with details on how to train and perform inference and stuff
- Prepare training on GPU:
    - Write notebook for testing GPU availability
- Implement generator models
- Augment dataset using generator models
- Train classifiers on augmented datasets
- Compare results

- Write tests for the classifier scripts ???
- Maybe create a class structure for representing the training data ?
- Add quantization ???
- Deployment of model
    - https://pytorch.org/serve/