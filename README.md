# gan-augmentation
Studying the utility of GANs and diffusion models in augmenting datasets used for classification problems

<br>

## Steps for enabling GPU access for PyTorch
- The following steps only apply to users having access to an Nvidia GPU
- Download the driver compatible with your graphics card and follow the prompts to complete the installation:
    - https://www.nvidia.com/en-us/drivers/
    - Choose the Download Type as Game Ready Driver (GRD)
    - Run `nvidia-smi` to check if the GPU supports CUDA. Remember the CUDA version used, if supported
- Download and install the CUDA toolkit:
    - https://developer.nvidia.com/cuda-toolkit-archive
    - Choose a version that matches or does not exceed your CUDA version
- Install the cuDNN library and move to the CUDA toolkit folder:
    - https://developer.nvidia.com/cudnn
    - Choose a version that matches your CUDA version
- Install the appropriate version of PyTorch:
    - https://pytorch.org/
    - Select your OS, programming language, and CUDA version, and copy the index url from the provided command
    - Uncomment the entry at the top of the `requirements.txt` file, and replace `<index_url>` with the copied index url: `--index-url <index_url>`
- If reporting GPU metrics in MLflow is also needed, uncommment the `pynvml` requirement in `requirements.txt`

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
- Run the `test_gpu_functionality.ipynb` notebook for testing whether a GPU is available, and if PyTorch has access to it
- Run the `train_classifier.ipynb` notebook for retraining any of the classifiers
- Run the `train_generator.ipynb` notebook for retraining any of the generators
- Run `nvidia-smi` to manually monitor the GPU utilization during training:
    - `nvidia-smi -l <n_seconds>` updates the output every `n_seconds`
    - `nvidia-smi -lms <n_milliseconds>` updates the output every `n_milliseconds`
- Open the MLflow UI at http://127.0.0.1:8080 to view the training results
- Alternatively, view the results in the `results/classifiers` and `results/generators` directories

<br>

## Steps for testing the project
- Create a virtual environment for the project (or activate the one created for running the project)
- Install the testing dependencies: `pip install -r test-requirements.txt`
- Run the pytest unit test suit from the project base directory: `python -m pytest tests`
