from copy import copy
from typing import Dict, List

import mlflow
import torch
import torch.autograd as autograd
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
from tqdm import tqdm

from scripts import config, utils
from scripts.config import GeneratorDataset, NormalizationRange
from scripts.generators.abstract_generator import AbstractGenerator
from scripts.models.wgan_gp import Generator, Critic

LOGGER = utils.get_logger(__name__)


class WGAN_GP_Generator(AbstractGenerator):
    """ Class representing a generator for TorchVision datasets,
    using a Wasserstein GAN with Gradient Penalty (WGAN-GP) """
    def preprocess_dataset(self) -> None:
        """ Loads the specified dataset and preprocesses it by converting to channels-first torch.FloatTensor, and
        scaling the values to the [-1.0, 1.0] range. If the dataset is grayscale, the channel dimension is squeezed in.
        The preprocessing is only applied when iterating over the dataset with a DataLoader. """
        if not self.preprocessed:
            self.hyperparams = copy(config.WGAN_GP_GEN_HYPERPARAMS)

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
        self.model = Generator(self.dataset_type).to(self.device, non_blocking=self.non_blocking)\
                                                 .apply(utils.initialize_weights)
        self.critic = Critic(self.dataset_type).to(self.device, non_blocking=self.non_blocking)\
                                                .apply(utils.initialize_weights)
        self.hyperparams = copy(config.WGAN_GP_GEN_HYPERPARAMS)

        if compute_batch_size:
            if self.device.type == "cuda":
                LOGGER.info(">>> Searching for the optimal batch size for this GPU and the Generator...")
                temp_generator = Generator(self.dataset_type).to(self.device, non_blocking=self.non_blocking)\
                                                             .apply(utils.initialize_weights)
                temp_critic = Critic(self.dataset_type).to(self.device, non_blocking=self.non_blocking)\
                                                        .apply(utils.initialize_weights)
                optimal_batch_size = utils.get_maximum_generator_batch_size(
                                        temp_generator,
                                        temp_critic,
                                        self.device,
                                        gen_input_shape=self.model.z_dim,
                                        disc_input_shape=self.batch_shape[1:],
                                        dataset_size=len(self.train_dataloader.dataset),
                                        max_batch_size=4096
                                    )

                self.hyperparams["BATCH_SIZE"] = optimal_batch_size
                del temp_generator, temp_critic
            else:
                LOGGER.info(">>> GPU not available, batch size computation skipped.")

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
        Uses RMSprop optimizer with lower learning rate for both the critic and generator.
        The loss function is the Wasserstein loss with gradient penalty to enforce Lipschitz constraint.
        
        Arguments:
            run_description (str): The description of the current run
            unnormalize (bool, optional): Whether to unnormalize the images for visualization
            normalization_range (NormalizationRange, optional): The normalization range used
        """
        # Define the optimizer - RMSprop is recommended for WGAN
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), 
                                                    lr=self.hyperparams["LEARNING_RATE"])
        self.gen_optimizer = torch.optim.RMSprop(self.model.parameters(), 
                                                 lr=self.hyperparams["LEARNING_RATE"])

        self._create_current_run_directory()

        # Keep track of metrics for evaluation
        self.training_history: Dict[str, List[float]] = {
            "critic_loss": [],
            "generator_loss": [],
            "gradient_penalty": []
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
            self.critic.train()
            self.model.train()

            for epoch in range(1, self.hyperparams["NUM_EPOCHS"] + 1):
                critic_loss = 0.0
                generator_loss = 0.0
                gradient_penalty_sum = 0.0

                for X_batch, y_batch in tqdm(self.train_dataloader):
                    X_batch = X_batch.to(self.device, non_blocking=self.non_blocking)
                    y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)

                    batch_size = X_batch.size(0)

                    # Prepare labels for critic input
                    channel_labels = y_batch[:, :, None, None]
                    channel_labels = channel_labels.repeat(1, 1, self.dataset_shape[1], self.dataset_shape[2])

                    # Train the critic multiple times per generator update
                    for _ in range(self.hyperparams["CRITIC_ITERATIONS"]):
                        self.critic_optimizer.zero_grad()

                        # Generate fake images
                        noise = torch.randn((batch_size, self.model.z_dim), device=self.device)
                        fake_images = self.model(noise, y_batch)

                        # Critic loss on real and fake images
                        critic_real = self.critic(X_batch, channel_labels).mean()
                        critic_fake = self.critic(fake_images.detach(), channel_labels).mean()
                        
                        # Compute gradient penalty
                        gp = self._gradient_penalty(X_batch, fake_images.detach(), channel_labels)
                        
                        # WGAN-GP loss: maximize critic_real - critic_fake - lambda*gp
                        batch_critic_loss = critic_fake - critic_real + self.hyperparams["LAMBDA_GP"] * gp
                        batch_critic_loss.backward()
                        self.critic_optimizer.step()

                        critic_loss += batch_critic_loss.item()
                        gradient_penalty_sum += gp.item()

                    # Train the generator
                    self.gen_optimizer.zero_grad()
                    
                    # Generate new fake images
                    noise = torch.randn((batch_size, self.model.z_dim), device=self.device)
                    fake_images = self.model(noise, y_batch)
                    
                    # Generator loss
                    gen_critic_output = self.critic(fake_images, channel_labels).mean()
                    batch_gen_loss = -gen_critic_output  # Minimize -critic(G(z))
                    batch_gen_loss.backward()
                    self.gen_optimizer.step()

                    generator_loss += batch_gen_loss.item()

                # Adjust for number of batches and critic iterations
                n_batches = len(self.train_dataloader)
                critic_loss /= (n_batches * self.hyperparams["CRITIC_ITERATIONS"])
                gradient_penalty_sum /= (n_batches * self.hyperparams["CRITIC_ITERATIONS"])
                generator_loss /= n_batches

                self.training_history["critic_loss"].append(critic_loss)
                self.training_history["gradient_penalty"].append(gradient_penalty_sum)
                self.training_history["generator_loss"].append(generator_loss)

                # Log the loss values to MLflow and console
                mlflow.log_metric("critic_loss", self.training_history["critic_loss"][-1], step=epoch)
                mlflow.log_metric("gradient_penalty", self.training_history["gradient_penalty"][-1], step=epoch)
                mlflow.log_metric("generator_loss", self.training_history["generator_loss"][-1], step=epoch)

                LOGGER.info(f"Epoch: {epoch}/{self.hyperparams['NUM_EPOCHS']}")
                LOGGER.info(f"> critic loss: {self.training_history['critic_loss'][-1]}")
                LOGGER.info(f"> gradient penalty: {self.training_history['gradient_penalty'][-1]}")
                LOGGER.info(f"> generator loss: {self.training_history['generator_loss'][-1]}")

                # Plot a batch of real and fake images
                noise = torch.randn((y_batch.shape[0], self.model.z_dim), device=self.device)
                fake = self.model(noise, y_batch)

                if unnormalize:
                    fake = utils.unnormalize_image(fake, normalization_range)
                    X_batch = utils.unnormalize_image(X_batch, normalization_range)

                LOGGER.info("> Real images:")
                self.display_image_batch(X_batch)
                LOGGER.info("> Fake images:")
                self.display_image_batch(fake)

        LOGGER.info(self.training_history)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by computing the Frechet Inception Distance between the generator
        distribution and real images distribution, on a small subdataset of 10000 samples. """
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
                self.model.eval()

                for X_batch, y_batch in tqdm(self.test_dataloader):
                    real = X_batch.to(self.device, non_blocking=self.non_blocking)
                    y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)

                    noise = torch.randn((y_batch.shape[0], self.model.z_dim), device=self.device)
                    fake = self.model(noise, y_batch)

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

    def _gradient_penalty(self, real_images: torch.Tensor, fake_images: torch.Tensor, 
                          channel_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient penalty for WGAN-GP
        
        Arguments:
            real_images (torch.Tensor): Batch of real images
            fake_images (torch.Tensor): Batch of generated images
            channel_labels (torch.Tensor): Batch of labels in channel format
            
        Returns:
            torch.Tensor: The gradient penalty value
        """
        batch_size = real_images.size(0)
        
        # Create random interpolation factors for each image in the batch
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Create interpolated images between real and fake
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        # Calculate critic scores on interpolated images
        critic_interpolated = self.critic(interpolated, channel_labels)
        
        # Calculate gradients of critic scores with respect to interpolated images
        gradients = autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Flatten the gradients
        gradients = gradients.view(batch_size, -1)
        
        # Calculate gradient penalty: (||grad|| - 1)^2
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty

    def _one_hot_encode(self, y: int) -> torch.Tensor:
        """ One-hot encodes the given label.

        Arguments:
            y (int): the label to be encoded
        """
        result = torch.zeros(len(self.class_labels), dtype=torch.float)

        return result.scatter_(dim=0, index=torch.tensor(y), value=1)
