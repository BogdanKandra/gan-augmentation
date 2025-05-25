# TODO
### Cursor Output:
#### WGAN+GP:
I've successfully implemented the WGAN-GP architecture for your project. Here's what I did:
1. Created a new model file scripts/models/wgan_gp.py with:
    - A Generator class similar to DCGAN but optimized for WGAN-GP
    - A Critic class (not Discriminator) that outputs a scalar score without sigmoid activation
2. Implemented the WGAN_GP_Generator class in scripts/generators/wgan_gp_generator.py with:
    - Dataset preprocessing with normalization to [-1, 1] range
    - Model building with proper weight initialization
    - Training loop with gradient penalty implementation
    - Evaluation using FID metric
3. Updated the hyperparameters in config.py with WGAN-GP specific values:
    - Smaller batch size (64) to accommodate gradient penalty calculation
    - Multiple critic iterations per generator update (5)
    - Gradient penalty coefficient (lambda = 10.0)
    - Lower learning rate (0.00005) for stability
The key differentiating features of this implementation compared to standard GANs are:
    - Using a critic instead of a discriminator that outputs unbounded scores
    - No sigmoid in the final layer of the critic
    - RMSprop optimizer instead of Adam
    - Wasserstein loss: critic_real - critic_fake
    - Gradient penalty to enforce 1-Lipschitz constraint
    - Multiple critic updates per generator update

#### DDPM:
I've successfully implemented the Denoising Diffusion Probabilistic Model (DDPM) architecture for your project. Here's a summary of what was created:
1. Model Architecture (scripts/models/ddpm.py):
    - A U-Net model with skip connections that predicts noise at each step
    - Self-attention layers to capture long-range dependencies
    - Time embeddings using sinusoidal position encoding
    - Class conditioning through embeddings
    - Complete diffusion process with forward noising and reverse denoising
2. Generator Class (scripts/generators/ddpm_generator.py):
    - Follows the same interface as other generators in your project
    - Data preprocessing with normalization to [-1, 1] range
    - Training loop with Adam optimizer and learning rate scheduling
    - Gradient clipping to prevent instability
    - Periodic sample generation during training
    - Evaluation using Frechet Inception Distance (FID)
3. Configuration (updated scripts/config.py):
    - Added DDPM-specific hyperparameters:
        - Noise schedule parameters (beta_start, beta_end)
        - Architecture parameters (hidden_channels, time_emb_dim)
        - Training parameters (batch_size, learning_rate, etc.)
The key differences between DDPM and previous GAN architectures:
1. Training Process:
    - DDPMs train a single network to predict noise, unlike GANs with adversarial training
    - They use a fixed noise schedule to gradually add/remove noise
    - Loss is simply MSE between predicted and actual noise
2. Generation Process:
    - Starts from pure noise and iteratively denoises
    - More stable than GANs but slower during inference (requires multiple steps)
3. Architecture:
    - Uses a U-Net with skip connections rather than separate generator/discriminator
    - Requires time step conditioning

#### DDIM:
I've successfully implemented the Denoising Diffusion Implicit Model (DDIM) architecture for your project. Here's a summary of the implementation:
1. DDIM Model (scripts/models/ddim.py):
    - Extended the DDPM architecture with a more efficient sampling process
    - Implemented non-Markovian deterministic sampling that requires fewer steps
    - Added configurable stochasticity parameter (eta) for controlling sampling behavior
    - Maintained the same training process but with accelerated generation
2. Generator Class (scripts/generators/ddim_generator.py):
    - Created a DDIM generator that follows the project structure
    - Reused the same U-Net architecture from DDPM for noise prediction
    - Implemented the specialized sampling process during generation
    - Maintained the same training and evaluation methodology
3. Configuration (updated scripts/config.py):
    - Added DDIM-specific hyperparameters:
        - SAMPLING_ETA: Controls stochasticity (0.0 for deterministic sampling)
        - SAMPLING_STEPS: Reduced number of steps for faster generation (50 vs 1000)
        - Same training parameters as DDPM
Key differences between DDIM and DDPM:
1. Sampling Efficiency:
    - DDIM can generate high-quality samples with far fewer steps (50 vs 1000)
    - DDIM uses a non-Markovian process that allows skipping intermediate steps
2. Controllable Generation:
    - DDIM allows for deterministic sampling (η=0) or varying levels of stochasticity
    - The deterministic nature enables interpolation between latent spaces
3. Same Training Process:
    - Despite different sampling, DDIM uses the same training approach as DDPM
    - The same U-Net architecture is used for noise prediction

### Write Jupyter Notebooks:
- Notebook for running inference with a chosen generator model
- Notebook for running inference with a chosen classifier model

### Implement GANs:
- WGAN+GP

### Implement Diffusion Models:
- DDPM
- DDIM
- Implement both DMs as conditional DMs
- Loss functions?
- Use the Frechet Inception Distance (FID) and Inception Score (IS) eval metrics

### Project Milestones:
- Augment datasets using generator models
	- How exactly ??
- Train the 4 classifiers on augmented datasets
- Analyze results

### Miscellaneous:
- Add the Optional[type] typing hint to all optional method arguments
- Add the Union[type|type] typing hint to multiple typed arguments
- Write separate requirement files for CPU and GPU processing
- Solve the PyTorch GPU requirement problem in requirements.txt
- Also plot the percentages out of total data in confusion matrix (right under the number)
- Consider moving the display_model() method from the abstract class to each child class (if the behaviour is sufficiently different)
- Investigate why GPU data transfer is slow:
    - Comment out self.non_blocking ??
- Also periodically save a checkpoint while training the GAN?
- Write tests for the classifier scripts ???
- Also save result artifacts in MLflow instead of the `results` directory
- Save the best model when early stopping
- Use the PyTorch profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Also implement the CID Index (Creativity, Inheritance, Diversity) metric for generators:
    https://shuyueg.github.io/doc/AIPR2019.pdf
- Also implement Multi-scale Structural Similarity Index Measure (MS-SSIM) as generator metric?
- Also implement early stopping in GAN training?
    - Use FID as the early stopping criterion
    - Use this method - https://arxiv.org/html/2405.20987v1
- Integrate TensorBoard ???
- Add L2 regularization to the CNN classifier?    # l2 = regularizers.l2(config.L2_LOSS_LAMBDA_2)
- Maybe create a class structure for representing the training data ?
- Implement quantization? - https://huggingface.co/docs/transformers/quantization/overview

### TO STUDY:
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

### Deployment / inference of model ideas:
- TorchServe: https://pytorch.org/serve/
- Run inference with ONNX Runtime (Check third ONNX reference)
- torch.compile and torch.jit.script: https://discuss.pytorch.org/t/efficient-way-to-train-on-gpu-and-inference-on-cpu/185040
- Update README file with details on how to train, perform inference and other functionalities

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
