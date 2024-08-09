# gan-augmentation
Studying the utility of GANs and diffusion models in augmenting datasets used for classification problems

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
