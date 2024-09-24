# TODO
- Implement GANs:
    - Investigate why GPU data transfer is slow:
        - Comment out self.non_blocking ??

    - Also periodically save a checkpoint while training the GAN?

    - WGAN+GP

- Add the Optional[type] typing hint to all optional method arguments
- Add the Union[type|type] typing hint to multiple typed arguments
- Write separate requirement files for CPU and GPU processing
- Solve the PyTorch GPU requirement problem in requirements.txt
- Also plot the percentages out of total data in confusion matrix (right under the number)
- Consider moving the display_model() method from the abstract class to each child class (if the behaviour is sufficiently different)

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
	- 1 x Notebook (train_generator.ipynb) for training and evaluating generator models
		- User chooses generator type (VanillaGAN / DCGAN / DDPM / DDIM)
		- User chooses dataset (FashionMNIST / CIFAR-10)
	- 1 x Notebook for running inference with a chosen generator model
	- 1 x Notebook for running inference with a chosen classifier model
	- 1 x Notebook for testing GPU availability


- TODO MISC:
    - Study performance in deep learning:
        https://docs.nvidia.com/deeplearning/performance/index.html
    - Study the effects of weight decay in optimizers (and how to apply it to our models)
    - Study the effects of batch size on training:
        https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
        https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e
        https://wandb.ai/ayush-thakur/dl-question-bank/reports/What-s-the-Optimal-Batch-Size-to-Train-a-Neural-Network---VmlldzoyMDkyNDU
        https://arxiv.org/abs/1404.5997
        https://arxiv.org/abs/1609.04836
        https://arxiv.org/abs/1711.00489
    - Also save result artifacts in MLflow instead of the `results` directory
    - Save the best model when early stopping
    - Use the PyTorch profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    - Also implement the CID Index (Creativity, Inheritance, Diversity) metric for generators:
        https://shuyueg.github.io/doc/AIPR2019.pdf
    - Also implement Multi-scale Structural Similarity Index Measure (MS-SSIM) as generator metric?
    - Also implement early stopping in GAN training?
        - Use FID as the early stopping criterion
        - Use this method - https://arxiv.org/html/2405.20987v1

- Deployment / inference of model ideas:
    - TorchServe: https://pytorch.org/serve/
    - Run inference with ONNX Runtime (Check third ONNX reference)
    - torch.compile and torch.jit.script: https://discuss.pytorch.org/t/efficient-way-to-train-on-gpu-and-inference-on-cpu/185040
- Update README file with details on how to train, perform inference and other functionalities
- Integrate TensorBoard ???
- Add L2 regularization to the CNN classifier?    # l2 = regularizers.l2(config.L2_LOSS_LAMBDA_2)
- Maybe create a class structure for representing the training data ?
- Implement quantization? - https://huggingface.co/docs/transformers/quantization/overview
- Write tests for the classifier scripts ???

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
## Misc
- https://www.quora.com/Do-convolutional-neural-networks-learn-to-be-spatially-invariant-at-the-last-layer-of-the-network-fully-connected-layer-Convolution-layers-produce-spatially-equivariant-output-but-what-about-the-spatial-invariance
- https://towardsdatascience.com/translational-invariance-vs-translational-equivariance-f9fbc8fca63a
- https://pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/  (Training CNN on Fashion-MNIST)
- https://medium.com/@mjbhobe/classifying-fashion-with-a-keras-cnn-achieving-94-accuracy-part-2-a5bd7a4e7e5a  (Training CNN on Fashion-MNIST)

## PyTorch tutorials
- https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
- https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
- https://www.kaggle.com/code/adrynh/pytorch-tutorial-with-fashion-mnist
- https://www.learnpytorch.io/
- https://discuss.pytorch.org/t/what-is-the-difference-between-creating-a-validation-set-using-random-split-as-opposed-to-subsetrandomsampler/72462

## Transfer Learning
- https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## TorchEval
- https://pytorch.org/torcheval/main/metric_example.html

## ONNX export
- https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
- https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
- https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

## MLflow
- https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
- https://mlflow.org/docs/latest/system-metrics/index.html
- https://towardsdatascience.com/5-tips-for-mlflow-experiment-tracking-c70ae117b03f

## GPU training
- https://huggingface.co/docs/transformers/model_memory_anatomy
- https://huggingface.co/docs/transformers/perf_train_gpu_one
- https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1
- https://wandb.ai/ayush-thakur/dl-question-bank/reports/How-To-Check-If-PyTorch-Is-Using-The-GPU--VmlldzoyMDQ0NTU
- https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
- https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu/48152675#48152675
- https://medium.com/@0429shen/cant-train-deep-learning-models-using-gpu-in-pytorch-even-with-a-graphics-card-f61505ed758e
- https://www.reddit.com/r/pytorch/comments/11izx0i/using_my_gpu_to_train/
- https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html (Memory Pinning)
- https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader  (<pin_memory> DataLoader argument usage)
- https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/  (Memory Pinning)
- https://www.kaggle.com/code/aisuko/memory-pinning-for-pytorch-dataloader  (Memory Pinning)
- https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work (<num_workers> DataLoader argument usage)

## GAN Evaluation
- https://www.sapien.io/blog/the-metrics-and-challenges-of-evaluating-generative-adversarial-networks-gans
- https://shuyueg.github.io/doc/AIPR2019.pdf  (CID Index metric)
- https://arxiv.org/abs/1802.03446  (Survey on GAN evaluation measures)
