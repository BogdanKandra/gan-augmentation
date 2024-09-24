from copy import copy
from typing import Dict, List

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
from tqdm import tqdm

from scripts import config, utils
from scripts.config import GeneratorDataset, NormalizationRange
from scripts.generators.abstract_generator import AbstractGenerator
from scripts.models.dcgan import Generator, Discriminator

LOGGER = utils.get_logger(__name__)


class DCGAN_Generator(AbstractGenerator):
    """ Class representing a generator for TorchVision datasets,
    using a Deep Convolutional Generative Adversarial Network (DCGAN) """
    def preprocess_dataset(self) -> None:
        """ Loads the specified dataset and preprocesses it by converting to channels-first torch.FloatTensor, and
        scaling the values to the [-1.0, 1.0] range. If the dataset is grayscale, the channel dimension is squeezed in.
        The preprocessing is only applied when iterating over the dataset with a DataLoader. """
        if not self.preprocessed:
            # Load the dataset and apply the preprocessing transform
            match self.dataset_type:
                case GeneratorDataset.FASHION_MNIST:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5,), std=(0.5,)),
                    ])
                    train_dataset = FashionMNIST(root="data",
                                                 train=True,
                                                 transform=transform,
                                                 target_transform=self._one_hot_encode,
                                                 download=True)
                    test_dataset = FashionMNIST(root="data",
                                                train=False,
                                                transform=transform,
                                                target_transform=self._one_hot_encode,
                                                download=True)
                case GeneratorDataset.CIFAR_10:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ])
                    train_dataset = CIFAR10(root="data",
                                            train=True,
                                            transform=transform,
                                            target_transform=self._one_hot_encode,
                                            download=True)
                    test_dataset = CIFAR10(root="data",
                                           train=False,
                                           transform=transform,
                                           target_transform=self._one_hot_encode,
                                           download=True)

            # Define the DataLoaders
            self.train_dataloader = DataLoader(dataset=train_dataset,
                                               batch_size=self.hyperparams["BATCH_SIZE"],
                                               shuffle=True,
                                               **self.dataloader_params)
            self.test_dataloader = DataLoader(dataset=test_dataset,
                                              batch_size=self.hyperparams["BATCH_SIZE"],
                                              **self.dataloader_params)

            self.preprocessed = True

    def build_model(self, compute_batch_size: bool = False) -> None:
        """ Defines the generator's model structure and stores it as an instance attribute.

        Arguments:
            compute_batch_size (bool, optional): whether to compute the maximum batch size for this model and device
        """
        self.model = Generator(self.dataset_type).to(self.device, non_blocking=self.non_blocking)\
                                                 .apply(utils.initialize_weights)
        self.discriminator = Discriminator(self.dataset_type).to(self.device, non_blocking=self.non_blocking)\
                                                             .apply(utils.initialize_weights)
        self.hyperparams = copy(config.DCGAN_GEN_HYPERPARAMS)

        if compute_batch_size:
            if self.device.type == "cuda":
                LOGGER.info(">>> Searching for the optimal batch size for this GPU and the Generator...")
                temp_generator = Generator(self.dataset_type).to(self.device, non_blocking=self.non_blocking)\
                                                             .apply(utils.initialize_weights)
                temp_discriminator = Discriminator(self.dataset_type).to(self.device, non_blocking=self.non_blocking)\
                                                                     .apply(utils.initialize_weights)
                optimal_batch_size = utils.get_maximum_generator_batch_size(
                                        temp_generator,
                                        temp_discriminator,
                                        self.device,
                                        gen_input_shape=self.model.z_dim,
                                        gen_output_shape=self.batch_shape[1:],
                                        disc_input_shape=self.batch_shape[1:],
                                        disc_output_shape=1,
                                        dataset_size=len(self.train_dataloader),
                                        max_batch_size=1024
                                        )
                self.hyperparams["BATCH_SIZE"] = optimal_batch_size
                del temp_generator, temp_discriminator
            else:
                LOGGER.info(">>> GPU not available, batch size computation skipped.")

    def train_model(self, run_description: str) -> None:
        """ Defines the training parameters and runs the training loop for the model currently in memory. Adam is used
        as the optimizer for both the discriminator and the generator, and the loss function to be optimised is the
        Binary Cross Entropy loss.

        Arguments:
            run_description (str): The description of the current run
        """
        # Define the optimizer and loss functions
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                               lr=self.hyperparams["LEARNING_RATE"],
                                               betas=(self.hyperparams["BETA_1"], self.hyperparams["BETA_2"]))
        self.gen_optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.hyperparams["LEARNING_RATE"],
                                              betas=(self.hyperparams["BETA_1"], self.hyperparams["BETA_2"]))
        self.loss = nn.BCEWithLogitsLoss()

        self._create_current_run_directory()

        # Keep track of metrics for evaluation
        self.training_history: Dict[str, List[float]] = {
            "discriminator_loss": [],
            "generator_loss": []
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
            self.discriminator.train()
            self.model.train()

            for epoch in range(1, self.hyperparams["NUM_EPOCHS"] + 1):
                discriminator_loss = 0.0
                generator_loss = 0.0

                for X_batch, y_batch in tqdm(self.train_dataloader):
                    X_batch = X_batch.to(self.device, non_blocking=self.non_blocking)
                    y_batch = y_batch.to(self.device, non_blocking=self.non_blocking)

                    # Update the discriminator
                    self.disc_optimizer.zero_grad()
                    batch_disc_loss = self._disc_loss(X_batch, y_batch)
                    batch_disc_loss.backward(retain_graph=True)
                    self.disc_optimizer.step()

                    # Update the generator
                    self.gen_optimizer.zero_grad()
                    batch_gen_loss = self._gen_loss(y_batch)
                    batch_gen_loss.backward()
                    self.gen_optimizer.step()

                    discriminator_loss += batch_disc_loss.item()
                    generator_loss += batch_gen_loss.item()

                self.training_history["discriminator_loss"].append(discriminator_loss)
                self.training_history["generator_loss"].append(generator_loss)

                # Log the loss values to MLflow and console
                mlflow.log_metric("discriminator_loss", self.training_history["discriminator_loss"][-1], step=epoch)
                mlflow.log_metric("generator_loss", self.training_history["generator_loss"][-1], step=epoch)

                LOGGER.info(f"Epoch: {epoch}/{self.hyperparams['NUM_EPOCHS']}")
                LOGGER.info(f"> discriminator loss: {self.training_history['discriminator_loss'][-1]}")
                LOGGER.info(f"> generator loss: {self.training_history['generator_loss'][-1]}")

                # Plot a batch of real and fake images
                noise = torch.randn((y_batch.shape[0], self.model.z_dim), device=self.device)
                fake = self.model(noise, y_batch)
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
                    # real = utils.unnormalize_image(real, NormalizationRange.TANGENT)
                    # fake = utils.unnormalize_image(fake, NormalizationRange.TANGENT)

                    fid.update(real, is_real=True)
                    fid.update(fake, is_real=False)

                self.evaluation_results["frechet-inception-distance"] = fid.compute().item()
                mlflow.log_metric("frechet-inception-distance", self.evaluation_results["frechet-inception-distance"])

        LOGGER.info(self.evaluation_results)

    def _gen_loss(self, one_hot_labels: torch.Tensor) -> torch.Tensor:
        """ Computes the generator loss for a batch of fake images.

        Arguments:
            one_hot_labels (torch.Tensor): A batch of one-hot labels

        Returns:
            torch.Tensor: A Torch scalar representing the loss value
        """
        # Generate a batch of fake images
        noise = torch.randn((one_hot_labels.shape[0], self.model.z_dim), device=self.device)
        fake = self.model(noise, one_hot_labels)

        # Compute the discriminator's predictions of the fake images
        channel_labels = one_hot_labels[:, :, None, None]
        channel_labels = channel_labels.repeat(1, 1, self.dataset_shape[1], self.dataset_shape[2])
        predictions = self.discriminator(fake, channel_labels)

        # Compute the loss for the fake images
        fake_labels = torch.ones_like(predictions)
        loss = self.loss(predictions, fake_labels)

        return loss

    def _disc_loss(self, real: torch.Tensor, one_hot_labels: torch.Tensor) -> torch.Tensor:
        """ Computes the discriminator loss for a batch of real images and a batch of fake images.

        Arguments:
            real (torch.Tensor): A batch of real images
            one_hot_labels (torch.Tensor): A batch of one-hot labels

        Returns:
            torch.Tensor: A Torch scalar representing the loss value
        """
        # Generate a batch of fake images
        noise = torch.randn((one_hot_labels.shape[0], self.model.z_dim), device=self.device)
        fake = self.model(noise, one_hot_labels)

        # Compute the discriminator's predictions of the fake images
        channel_labels = one_hot_labels[:, :, None, None]
        channel_labels = channel_labels.repeat(1, 1, self.dataset_shape[1], self.dataset_shape[2])
        predictions = self.discriminator(fake.detach(), channel_labels)

        # Compute the loss for the fake images
        true_labels = torch.zeros_like(predictions)
        loss_fakes = self.loss(predictions, true_labels)

        # Compute the discriminator's predictions of the real images
        predictions = self.discriminator(real, channel_labels)

        # Compute the loss for the real images
        true_labels = torch.ones_like(predictions)
        loss_reals = self.loss(predictions, true_labels)

        # Combine the losses
        disc_loss = (loss_fakes + loss_reals) / 2

        return disc_loss

    def _one_hot_encode(self, y: int) -> torch.Tensor:
        """ One-hot encodes the given label.

        Arguments:
            y (int): the label to be encoded
        """
        result = torch.zeros(len(self.class_labels), dtype=torch.float)

        return result.scatter_(dim=0, index=torch.tensor(y), value=1)
