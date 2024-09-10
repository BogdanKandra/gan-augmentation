from copy import copy
from typing import Dict, List

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import FrechetInceptionDistance
from tqdm import tqdm

from scripts import config, utils
from scripts.generators.torchvision_dataset_generator import TorchVisionDatasetGenerator
from scripts.models.gan import Generator, Discriminator

LOGGER = utils.get_logger(__name__)


class GANGenerator(TorchVisionDatasetGenerator):
    """ Class representing a generator for TorchVision datasets,
    using a vanilla Generative Adversarial Network (GAN) """
    def preprocess_dataset(self) -> None:
        """ Preprocesses the dataset currently in memory by reshaping it and scaling the values. """
        if not self.preprocessed:
            if len(self.X.shape) == 3:
                # If the loaded dataset is grayscale, add the channel dimension
                self.X = torch.unsqueeze(self.X, dim=1)
            elif len(self.X.shape) == 4 and self.X.shape[3] == 3:
                # If the loaded dataset is RGB channels-last, transform to channel-first
                self.X = self.X.permute(0, 3, 1, 2)

            # Convert to float and scale to [0, 1]
            self.X = self.X.to(torch.float32) / 255.0

            self.preprocessed = True

    def build_model(self, compute_batch_size: bool = False) -> None:
        """ Defines the generator's model structure and stores it as an instance attribute.

        Arguments:
            compute_batch_size (bool, optional): whether to compute the maximum batch size for this model and device
        """
        self.model = Generator(self.dataset_type).to(self.device)
        self.discriminator = Discriminator(self.dataset_type).to(self.device)
        self.hyperparams = copy(config.GAN_GEN_HYPERPARAMS)

        if compute_batch_size:
            if self.device.type == 'cuda':
                LOGGER.info('>>> Searching for the optimal batch size for this GPU and the Generator...')
                temp_generator = Generator(self.dataset_type).to(self.device)
                temp_discriminator = Discriminator(self.dataset_type).to(self.device)
                optimal_batch_size = utils.get_maximum_generator_batch_size(temp_generator,
                                                                            temp_discriminator,
                                                                            self.device,
                                                                            gen_input_shape=self.model.z_dim,
                                                                            gen_output_shape=self.X.shape[1:],
                                                                            disc_input_shape=self.X.shape[1:],
                                                                            disc_output_shape=1,
                                                                            dataset_size=self.X.shape[0],
                                                                            max_batch_size=512)
                self.hyperparams['BATCH_SIZE'] = optimal_batch_size
                del temp_generator, temp_discriminator
            else:
                LOGGER.info('>>> GPU not available, batch size computation skipped.')

    def train_model(self, run_description: str) -> None:
        """ Defines the training parameters and runs the training loop for the model currently in memory. Adam is used
        as the optimizer for both the discriminator and the generator, the loss function to be optimised is the Binary
        Cross Entropy loss, and the loss is measured. An early stopping mechanism is used to prevent overfitting.

        Arguments:
            run_description (str): The description of the current run
        """
        # Define the optimizer and loss functions
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hyperparams['LEARNING_RATE'])
        self.gen_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['LEARNING_RATE'])
        self.loss = nn.BCEWithLogitsLoss()

        self._create_current_run_directory()

        # Keep track of metrics for evaluation
        self.training_history: Dict[str, List[float]] = {
            "discriminator_loss": [],
            "generator_loss": []
        }
        early_stopping_counter = 0

        # Setup and start an MLflow run
        mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
        experiment_name = f"{self.__class__.__name__} {self.dataset_type.name}"
        mlflow.set_experiment(experiment_name)
        run_name = ' '.join(str(s) for s in self.results_subdirectory.split(' ')[2:])

        with mlflow.start_run(run_name=run_name, description=run_description, log_system_metrics=True) as run:
            # Define the DataLoader
            dataset = TensorDataset(self.X, self.y)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.hyperparams['BATCH_SIZE'],
                                    shuffle=True,
                                    pin_memory=self.pin_memory,
                                    pin_memory_device=self.pin_memory_device)

            # Log the hyperparameters to MLflow
            mlflow.log_params(self.hyperparams)

            # Run the training loop
            self.discriminator.train()
            self.model.train()

            for epoch in range(1, self.hyperparams['NUM_EPOCHS'] + 1):
                discriminator_loss = 0.0
                generator_loss = 0.0

                for batch, _ in tqdm(dataloader):
                    batch = batch.to(self.device)

                    # Update the discriminator
                    self.disc_optimizer.zero_grad()
                    batch_disc_loss = self._disc_loss(batch)
                    batch_disc_loss.backward(retain_graph=True)
                    self.disc_optimizer.step()

                    # Update the generator
                    self.gen_optimizer.zero_grad()
                    batch_gen_loss = self._gen_loss()
                    batch_gen_loss.backward(retain_graph=True)
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

                # # Early stopping
                # best_loss = min(self.training_history["generator_loss"])
                # if self.training_history["generator_loss"][-1] <= best_loss:
                #     if early_stopping_counter != 0:
                #         early_stopping_counter = 0
                #         LOGGER.info(">> Early stopping counter reset.")
                #     self.best_model_state = self.model.state_dict()
                # else:
                #     early_stopping_counter += 1
                #     LOGGER.info(f">> Early stopping counter increased to {early_stopping_counter}.")

                # if early_stopping_counter == self.hyperparams['EARLY_STOPPING_TOLERANCE']:
                #     LOGGER.info(">> Training terminated due to early stopping!")
                #     break

            self.run_id = run.info.run_id

        LOGGER.info(self.training_history)

    def evaluate_model(self) -> None:
        """ Evaluates the model currently in memory by computing the Frechet Inception Distance of the generator. """
        # Define the FID evaluation metric
        fid = FrechetInceptionDistance(device=self.device)

        self.evaluation_results: Dict[str, float] = {
            "frechet-inception-distance": 0.0
        }

        # Setup and start an MLflow run
        mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
        experiment_name = f"{self.__class__.__name__} {self.dataset_type.name}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_id=self.run_id, log_system_metrics=True):
            # Define the DataLoader
            dataset = TensorDataset(self.X, self.y)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.hyperparams['BATCH_SIZE'],
                                    pin_memory=self.pin_memory,
                                    pin_memory_device=self.pin_memory_device)

            # Gradient computation is not required during evaluation
            with torch.no_grad():
                self.model.eval()

                for batch, _ in tqdm(dataloader):
                    real = batch.to(self.device)
                    noise = torch.randn((self.hyperparams['BATCH_SIZE'], self.model.z_dim), device=self.device)
                    fake = self.model(noise)

                    # The Inception-V3 model used for computing FID expects 3-channel images
                    if real.shape[1] == 1:
                        real = torch.cat([real] * 3, dim=1)
                        fake = torch.cat([fake] * 3, dim=1)

                    fid.update(real, is_real=True)
                    fid.update(fake, is_real=False)

                self.evaluation_results["frechet-inception-distance"] = fid.compute().item()
                mlflow.log_metric("frechet-inception-distance", self.evaluation_results["frechet-inception-distance"])

        LOGGER.info(self.evaluation_results)

    def _gen_loss(self) -> torch.Tensor:
        """ Computes the generator loss for a batch of fake images.

        Returns:
            torch.Tensor: A Torch scalar representing the loss value
        """
        # Generate noise for the generator
        noise = torch.randn((self.hyperparams['BATCH_SIZE'], self.model.z_dim), device=self.device)

        # Generate a batch of fake images
        fake = self.model(noise)

        # Compute the discriminator's predictions of the fake images
        predictions = self.discriminator(fake)

        # Compute the loss for the fake images
        labels = torch.ones_like(predictions)
        loss = self.loss(predictions, labels)

        return loss

    def _disc_loss(self, real: torch.Tensor) -> torch.Tensor:
        """ Computes the discriminator loss for a batch of real images and a batch of fake images.

        Arguments:
            real (torch.Tensor): A batch of real images

        Returns:
            torch.Tensor: A Torch scalar representing the loss value
        """
        # Generate noise for the generator
        noise = torch.randn((self.hyperparams['BATCH_SIZE'], self.model.z_dim), device=self.device)

        # Generate a batch of fake images
        fake = self.model(noise)

        # Compute the discriminator's predictions of the fake images
        predictions = self.discriminator(fake.detach())

        # Compute the loss for the fake images
        labels = torch.zeros_like(predictions)
        loss_fakes = self.loss(predictions, labels)

        # Compute the discriminator's predictions of the fake images
        predictions = self.discriminator(real)

        # Compute the loss for the real images
        labels = torch.ones_like(predictions)
        loss_reals = self.loss(predictions, labels)

        # Combine the losses
        disc_loss = (loss_fakes + loss_reals) / 2

        return disc_loss
