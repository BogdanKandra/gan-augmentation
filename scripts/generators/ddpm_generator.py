from copy import copy
from typing import Dict, List

import mlflow
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
from tqdm import tqdm

from scripts import config, utils
from scripts.config import GeneratorDataset, NormalizationRange
from scripts.generators.abstract_generator import AbstractGenerator
from scripts.models.ddpm import Unet, DiffusionModel

LOGGER = utils.get_logger(__name__)


class DDPM_Generator(AbstractGenerator):
    """ Class representing a generator for TorchVision datasets,
    using a Denoising Diffusion Probabilistic Model (DDPM) """
    def preprocess_dataset(self) -> None:
        """ Loads the specified dataset and preprocesses it by converting to channels-first torch.FloatTensor, and
        scaling the values to the [-1.0, 1.0] range. If the dataset is grayscale, the channel dimension is squeezed in.
        The preprocessing is only applied when iterating over the dataset with a DataLoader. """
        if not self.preprocessed:
            self.hyperparams = copy(config.DDPM_GEN_HYPERPARAMS)

            # Load the dataset and apply the preprocessing transform
            match self.dataset_type:
                case GeneratorDataset.FASHION_MNIST:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,)),
                    ])
                    self.train_dataset = FashionMNIST(root="data",
                                                      train=True,
                                                      transform=transform,
                                                      target_transform=self._one_hot_encode,
                                                      download=True)
                    self.test_dataset = FashionMNIST(root="data",
                                                     train=False,
                                                     transform=transform,
                                                     target_transform=self._one_hot_encode,
                                                     download=True)
                case GeneratorDataset.CIFAR_10:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ])
                    self.train_dataset = CIFAR10(root="data",
                                                 train=True,
                                                 transform=transform,
                                                 target_transform=self._one_hot_encode,
                                                 download=True)
                    self.test_dataset = CIFAR10(root="data",
                                                train=False,
                                                transform=transform,
                                                target_transform=self._one_hot_encode,
                                                download=True)

            self.preprocessed = True

    def build_model(self, compute_batch_size: bool = False) -> None:
        """ Defines the generator's model structure and stores it as an instance attribute.

        Arguments:
            compute_batch_size (bool, optional): whether to compute the maximum batch size for this model and device
        """
        # Initialize UNet noise predictor
        noise_predictor = Unet(
            dataset=self.dataset_type,
            hidden_channels=self.hyperparams["HIDDEN_CHANNELS"],
            time_emb_dim=self.hyperparams["TIME_EMB_DIM"],
            num_class_emb=len(self.class_labels),
            dropout=self.hyperparams["DROPOUT"]
        ).to(self.device, non_blocking=self.non_blocking)
        
        # Initialize the diffusion model
        self.model = DiffusionModel(
            noise_predictor=noise_predictor,
            device=self.device,
            timesteps=self.hyperparams["TIMESTEPS"],
            beta_start=self.hyperparams["BETA_START"],
            beta_end=self.hyperparams["BETA_END"]
        )

        if compute_batch_size:
            if self.device.type == "cuda":
                LOGGER.info(">>> GPU detected, using a conservative batch size due to DDPM memory requirements.")
                # DDPM is memory intensive, so we use a smaller batch size
                optimal_batch_size = min(64, self.hyperparams["BATCH_SIZE"])
                self.hyperparams["BATCH_SIZE"] = optimal_batch_size
            else:
                LOGGER.info(">>> CPU detected, using a small batch size.")
                self.hyperparams["BATCH_SIZE"] = 16

        # Define train and test DataLoaders
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hyperparams["BATCH_SIZE"],
                                           shuffle=True,
                                           **self.dataloader_params)
        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.hyperparams["BATCH_SIZE"],
                                          **self.dataloader_params)

    def train_model(self,
                    run_description: str,
                    unnormalize: bool = False,
                    normalization_range: NormalizationRange = None) -> None:
        """ Defines the training parameters and runs the training loop for the model currently in memory. 
        Uses Adam optimizer with learning rate scheduling for the U-Net noise predictor.
        The loss function is the mean squared error between predicted and actual noise.
        
        Arguments:
            run_description (str): The description of the current run
            unnormalize (bool, optional): Whether to unnormalize the images for visualization
            normalization_range (NormalizationRange, optional): The normalization range used
        """
        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.model.noise_predictor.parameters(), 
            lr=self.hyperparams["LEARNING_RATE"],
            betas=(self.hyperparams["BETA_1"], self.hyperparams["BETA_2"])
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hyperparams["LEARNING_RATE"],
            total_steps=self.hyperparams["NUM_EPOCHS"] * len(self.train_dataloader),
            pct_start=0.1
        )

        self._create_current_run_directory()

        # Keep track of metrics for evaluation
        self.training_history: Dict[str, List[float]] = {
            "loss": []
        }

        # Setup and start an MLflow run
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        experiment_name = f"{self.__class__.__name__} {self.dataset_type.name}"
        mlflow.set_experiment(experiment_name)
        run_name = " ".join(str(s) for s in self.results_subdirectory.split(" ")[2:])

        with mlflow.start_run(run_name=run_name, description=run_description, log_system_metrics=True) as run:
            self.run_id = run.info.run_id

            # Log the hyperparameters to MLflow
            mlflow.log_params(self.hyperparams)

            # Run the training loop
            self.model.noise_predictor.train()

            for epoch in range(1, self.hyperparams["NUM_EPOCHS"] + 1):
                epoch_loss = 0.0

                for X_batch, y_batch in tqdm(self.train_dataloader):
                    X_batch = X_batch.to(self.device, non_blocking=self.non_blocking)
                    y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)
                    
                    batch_size = X_batch.size(0)
                    
                    # Sample random timesteps for each image
                    t = torch.randint(0, self.model.timesteps, (batch_size,), device=self.device, dtype=torch.long)
                    
                    # Forward pass and compute loss
                    optimizer.zero_grad()
                    loss = self.model.p_losses(X_batch, t, y_batch)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.noise_predictor.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()

                # Average loss for the epoch
                epoch_loss /= len(self.train_dataloader)
                self.training_history["loss"].append(epoch_loss)

                # Log the loss values to MLflow and console
                mlflow.log_metric("loss", self.training_history["loss"][-1], step=epoch)
                LOGGER.info(f"Epoch: {epoch}/{self.hyperparams['NUM_EPOCHS']}")
                LOGGER.info(f"> loss: {self.training_history['loss'][-1]}")

                # Generate and plot sample images every few epochs
                if epoch % self.hyperparams["SAMPLE_INTERVAL"] == 0 or epoch == self.hyperparams["NUM_EPOCHS"]:
                    LOGGER.info("> Generating sample images...")
                    
                    # Get a batch of labels for generation
                    sample_labels = y_batch[:8]  # Use first 8 labels from the last batch
                    
                    # Generate images
                    with torch.no_grad():
                        self.model.noise_predictor.eval()
                        generated_images = self.model.sample(
                            batch_size=8,
                            image_channels=self.dataset_shape[0],
                            image_size=self.dataset_shape[1],
                            labels=sample_labels
                        )
                        self.model.noise_predictor.train()
                    
                    if unnormalize:
                        generated_images = utils.unnormalize_image(generated_images, normalization_range)
                    
                    # Display the generated images
                    LOGGER.info("> Generated images:")
                    self.display_image_batch(generated_images)
                    
                    # Save the model checkpoint
                    torch.save(
                        self.model.noise_predictor.state_dict(),
                        f"{self.results_subdirectory}/checkpoint_epoch_{epoch}.pt"
                    )

        LOGGER.info(self.training_history)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by computing the Frechet Inception Distance between the generator
        distribution and real images distribution. """
        # Define the FID evaluation metric
        fid = FrechetInceptionDistance(device=self.device)

        self.evaluation_results: Dict[str, float] = {
            "frechet-inception-distance": 0.0
        }

        # Setup and start an MLflow run
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        experiment_name = f"{self.__class__.__name__} {self.dataset_type.name}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_id=self.run_id, log_system_metrics=True):
            # Gradient computation is not required during evaluation
            with torch.no_grad():
                self.model.noise_predictor.eval()

                for X_batch, y_batch in tqdm(self.test_dataloader):
                    real = X_batch.to(self.device, non_blocking=self.non_blocking)
                    y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)

                    # Generate samples with the same labels
                    fake = self.model.sample(
                        batch_size=real.shape[0],
                        image_channels=self.dataset_shape[0],
                        image_size=self.dataset_shape[1],
                        labels=y_batch
                    )

                    # The Inception-V3 model used for computing FID expects 3-channel images
                    if real.shape[1] == 1:
                        real = torch.cat([real] * 3, dim=1)
                        fake = torch.cat([fake] * 3, dim=1)

                    # The FID expects images in the range [0, 1]
                    # Converting from [-1, 1] to [0, 1] range
                    real = (real + 1) / 2
                    fake = (fake + 1) / 2

                    fid.update(real, is_real=True)
                    fid.update(fake, is_real=False)

                self.evaluation_results["frechet-inception-distance"] = fid.compute().item()
                mlflow.log_metric("frechet-inception-distance", self.evaluation_results["frechet-inception-distance"])

        LOGGER.info(self.evaluation_results)

    def _one_hot_encode(self, y: int) -> torch.Tensor:
        """ One-hot encodes the given label.

        Arguments:
            y (int): the label to be encoded
        """
        result = torch.zeros(len(self.class_labels), dtype=torch.float)

        return result.scatter_(dim=0, index=torch.tensor(y), value=1)
