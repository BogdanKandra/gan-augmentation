import torch
import torch.nn.functional as F
from torch import Tensor

from scripts.models.ddpm import Unet, DiffusionModel


class DDIMModel(DiffusionModel):
    """
    DDIM (Denoising Diffusion Implicit Model) implementation.
    Extends the DDPM model but uses a deterministic, non-Markovian sampling process
    for faster inference with fewer timesteps.
    """
    def __init__(
        self, 
        noise_predictor: Unet, 
        device: torch.device,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        ddim_sampling_eta: float = 0.0,  # η=0 for deterministic sampling
        ddim_sampling_steps: int = 50    # Number of steps for DDIM sampling
    ) -> None:
        """
        Initialize the DDIM model.
        
        Args:
            noise_predictor: U-Net model for noise prediction
            device: Device to run the model on
            timesteps: Number of diffusion steps for training
            beta_start: Starting variance schedule value
            beta_end: Ending variance schedule value
            ddim_sampling_eta: Controls the stochasticity of the sampling (0 = deterministic)
            ddim_sampling_steps: Number of timesteps to use for DDIM sampling
        """
        super().__init__(
            noise_predictor=noise_predictor,
            device=device,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        self.ddim_sampling_eta = ddim_sampling_eta
        self.ddim_sampling_steps = ddim_sampling_steps
        
        # Calculate the subset of timesteps to use for DDIM
        self.ddim_timesteps = self._get_ddim_timesteps()
        
    def _get_ddim_timesteps(self) -> torch.Tensor:
        """
        Get the subset of timesteps to use for DDIM sampling.
        
        Returns:
            Tensor of timesteps to use for DDIM
        """
        # Select evenly spaced timesteps from the original diffusion process
        step_size = self.timesteps // self.ddim_sampling_steps
        ddim_timesteps = torch.arange(0, self.timesteps, step_size, device=self.device)
        
        # Make sure we include the final timestep
        if ddim_timesteps[-1] != self.timesteps - 1:
            ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([self.timesteps - 1], device=self.device)])
            
        # Reverse for sampling (start from noise and go to clean image)
        return torch.flip(ddim_timesteps, dims=[0])
    
    def _get_prev_timestep(self, t_index: int) -> torch.Tensor:
        """
        Get the previous timestep in the DDIM sampling process.
        
        Args:
            t_index: Current index in the DDIM timesteps array
            
        Returns:
            Previous timestep value
        """
        if t_index == len(self.ddim_timesteps) - 1:
            return torch.tensor([0], device=self.device)
        else:
            return self.ddim_timesteps[t_index + 1]
    
    @torch.no_grad()
    def ddim_sample(self, x_t: Tensor, t_index: int, t_cur: Tensor, labels: Tensor) -> Tensor:
        """
        Single step of the DDIM sampling process.
        
        Args:
            x_t: Current noisy image
            t_index: Index in the DDIM timesteps array
            t_cur: Current timestep tensor
            labels: Class labels tensor
            
        Returns:
            Sample for next timestep
        """
        # Get the current and previous timestep values
        t_prev = self._get_prev_timestep(t_index)
        
        # Predict noise
        predicted_noise = self.noise_predictor(x_t, t_cur, labels)
        
        # Get alpha values for current and previous timesteps
        alpha_cumprod_t = self.alphas_cumprod[t_cur]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)
        
        # Reshape alpha values for broadcasting
        alpha_cumprod_t = alpha_cumprod_t.view(-1, 1, 1, 1)
        alpha_cumprod_prev = alpha_cumprod_prev.view(-1, 1, 1, 1)
        
        # Predict x_0 (clean image)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_cumprod_t)
        predicted_x0 = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_cumprod_t
        
        # DDIM update formula
        # Direction pointing to x_t
        direction_to_xt = torch.sqrt(1 - alpha_cumprod_prev - self.ddim_sampling_eta**2 * (1 - alpha_cumprod_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)) * predicted_noise
        
        # Random noise for stochastic sampling (only if eta > 0)
        random_noise = torch.zeros_like(x_t)
        if self.ddim_sampling_eta > 0:
            sigma = self.ddim_sampling_eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev))
            random_noise = torch.randn_like(x_t) * sigma.view(-1, 1, 1, 1)
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_cumprod_prev) * predicted_x0 + direction_to_xt + random_noise
        
        return x_prev
    
    @torch.no_grad()
    def ddim_sample_loop(self, shape: tuple, labels: Tensor) -> Tensor:
        """
        Full DDIM sampling loop to generate images from noise.
        
        Args:
            shape: Shape of the images to generate (batch_size, channels, height, width)
            labels: Class labels tensor
            
        Returns:
            Generated images
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=self.device)
        
        # Iteratively denoise using DDIM
        for i, timestep in enumerate(self.ddim_timesteps):
            # Create batch of current timesteps
            t_batch = torch.full((shape[0],), timestep, device=self.device, dtype=torch.long)
            x_t = self.ddim_sample(x_t, i, t_batch, labels)
        
        # Clamp values to valid image range
        return torch.clamp(x_t, -1., 1.)
    
    @torch.no_grad()
    def sample(self, batch_size: int, image_channels: int, image_size: int, labels: Tensor) -> Tensor:
        """
        Generate a batch of images from noise using DDIM sampling.
        
        Args:
            batch_size: Number of images to generate
            image_channels: Number of channels in the images
            image_size: Size of the images (assumed square)
            labels: Class labels tensor
            
        Returns:
            Generated images
        """
        return self.ddim_sample_loop(
            (batch_size, image_channels, image_size, image_size), 
            labels
        )
