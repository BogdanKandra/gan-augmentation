# gan-augmentation
Studying the utility of GANs and diffusion models in augmenting datasets used for classification problems

<br>

## Steps for running the project
- Create a virtual environment for the project: `conda create -n <env_name> python=3.10`
- Activate the virtual environment: `conda activate <env_name>`
- Open a terminal and CD to the project root directory
- Install the development dependencies: `pip install -r requirements.txt`
- Run any of the provided Jupyter notebooks, within the created environment

### Retraining models
- (Optional) Set the following environment variable, to avoid Git warnings from MLflow:
    - `export GIT_PYTHON_REFRESH=quiet` (On Linux)
    - `set GIT_PYTHON_REFRESH=quiet` (On Windows)
- Start a local MLflow tracking server (from the project root): `mlflow server --host 127.0.0.1 --port 8080`
- Run the `train_classifier.ipynb` notebook for retraining any of the classifiers
- Run the `train_generator.ipynb` notebook for retraining any of the generators
- Open the MLflow UI at http://127.0.0.1:8080 to view the training results
- Alternatively, view the results in the `results/classifiers` and `results/generators` directories

<br>

## Steps for testing the project
- Create a virtual environment for the project (or activate the one created for running the project)
- Install the testing dependencies: `pip install -r test-requirements.txt`
- Run the pytest unit test suit from the project base directory: `python -m pytest tests`
