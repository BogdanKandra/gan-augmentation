import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from scripts import config
from scripts.config import GeneratorDataset


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for noise level conditioning.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        """
        Compute sinusoidal embeddings for a batch of timesteps.
        
        Args:
            time: Tensor of shape (batch_size,)
            
        Returns:
            Tensor of shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    A ResNet-like block with optional self-attention.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        use_attention: bool = False
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        
        # Main convolution path
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.temb_proj = nn.Linear(temb_dim, out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
        
        # Self-attention layer
        if use_attention:
            self.attention = SelfAttention(out_channels)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        """
        Forward pass through the block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            temb: Time embedding tensor of shape (batch_size, temb_dim)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Residual connection
        h = h + self.residual(x)
        
        # Self-attention
        if self.use_attention:
            h = self.attention(h)
            
        return h


class SelfAttention(nn.Module):
    """
    Self-attention module for capturing long-range dependencies.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the self-attention module.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels, height, width)
        """
        _, _, h, w = x.shape
        x_norm = self.norm(x)
        
        # Reshape for multi-head attention: (batch, channels, h, w) -> (batch, h*w, channels)
        x_flat = x_norm.flatten(2).permute(0, 2, 1)
        
        # Self-attention
        attn_output, _ = self.mha(x_flat, x_flat, x_flat)
        
        # Reshape back to original format: (batch, h*w, channels) -> (batch, channels, h, w)
        attn_output = attn_output.permute(0, 2, 1).view(x.shape[0], self.channels, h, w)
        
        # Residual connection
        return x + attn_output


class Unet(nn.Module):
    """
    U-Net model for noise prediction in diffusion models.
    """
    def __init__(
        self,
        dataset: GeneratorDataset,
        hidden_channels: int = 128,
        time_emb_dim: int = 256,
        num_class_emb: int = 10,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # Dataset parameters
        match dataset:
            case GeneratorDataset.FASHION_MNIST:
                self.in_channels = config.FASHION_MNIST_SHAPE[0]
                self.image_size = config.FASHION_MNIST_SHAPE[1]
                self.class_emb_dim = len(config.FASHION_MNIST_CLASS_LABELS)
            case GeneratorDataset.CIFAR_10:
                self.in_channels = config.CIFAR_10_SHAPE[0]
                self.image_size = config.CIFAR_10_SHAPE[1]
                self.class_emb_dim = len(config.CIFAR_10_CLASS_LABELS)
            case _:
                raise ValueError("Unsupported dataset type")
        
        # Dimensionality
        self.hidden_channels = hidden_channels
        self.time_emb_dim = time_emb_dim
        
        # Time and class embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Class conditioning projection
        self.class_embedding = nn.Embedding(num_class_emb, time_emb_dim)
        
        # Initial projection
        self.init_conv = nn.Conv2d(self.in_channels, hidden_channels, 3, padding=1)
        
        # Downsampling blocks
        down_channels = [hidden_channels, hidden_channels*2, hidden_channels*4, hidden_channels*8]
        self.downs = nn.ModuleList([])
        
        for i, channels in enumerate(down_channels):
            use_attention = (i >= 2)  # Apply attention in deeper layers
            if i == 0:
                self.downs.append(Block(hidden_channels, channels, time_emb_dim, use_attention))
            else:
                self.downs.append(Block(down_channels[i-1], channels, time_emb_dim, use_attention))
        
        # Middle blocks with attention
        self.mid_block1 = Block(down_channels[-1], down_channels[-1], time_emb_dim, True)
        self.mid_block2 = Block(down_channels[-1], down_channels[-1], time_emb_dim, False)
        
        # Upsampling blocks
        up_channels = [hidden_channels*8, hidden_channels*4, hidden_channels*2, hidden_channels]
        self.ups = nn.ModuleList([])
        
        for i, channels in enumerate(up_channels):
            use_attention = (i <= 1)  # Apply attention in deeper layers
            if i == len(up_channels) - 1:
                self.ups.append(Block(up_channels[i-1] + down_channels[0], channels, time_emb_dim, use_attention))
            else:
                self.ups.append(Block(up_channels[i-1] + down_channels[len(down_channels)-1-i], channels, time_emb_dim, use_attention))
        
        # Output layers
        self.final_norm = nn.GroupNorm(8, hidden_channels)
        self.final_conv = nn.Conv2d(hidden_channels, self.in_channels, 3, padding=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Noisy image tensor of shape (batch_size, channels, height, width)
            t: Timesteps tensor of shape (batch_size,)
            labels: Class labels tensor of shape (batch_size,) or one-hot encoding
            
        Returns:
            Predicted noise tensor of shape (batch_size, channels, height, width)
        """
        # Convert one-hot labels to indices if needed
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = torch.argmax(labels, dim=1)
            
        # Get time embeddings
        t_emb = self.time_mlp(t)
        
        # Get class embeddings and combine with time embeddings
        c_emb = self.class_embedding(labels)
        temb = t_emb + c_emb
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Downsampling path
        skip_connections = []
        for down in self.downs:
            h = down(h, temb)
            skip_connections.append(h)
            h = F.avg_pool2d(h, 2)
        
        # Middle blocks
        h = self.mid_block1(h, temb)
        h = self.mid_block2(h, temb)
        
        # Upsampling path
        for up, skip in zip(self.ups, reversed(skip_connections)):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            # Apply dropout to skip connections for regularization
            h = torch.cat([h, self.dropout(skip)], dim=1)
            h = up(h, temb)
        
        # Final layers
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        
        return h


class DiffusionModel:
    """
    Diffusion model implementation that handles the noise schedule and 
    sampling process for a Denoising Diffusion Probabilistic Model.
    """
    def __init__(
        self, 
        noise_predictor: Unet, 
        device: torch.device,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ) -> None:
        """
        Initialize the diffusion model.
        
        Args:
            noise_predictor: U-Net model for noise prediction
            device: Device to run the model on
            timesteps: Number of diffusion steps
            beta_start: Starting variance schedule value
            beta_end: Ending variance schedule value
        """
        self.noise_predictor = noise_predictor
        self.device = device
        self.timesteps = timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor = None) -> Tensor:
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_0: Clean images tensor of shape (batch_size, channels, height, width)
            t: Timesteps tensor of shape (batch_size,)
            noise: Optional noise tensor (generated if None)
            
        Returns:
            Noisy images tensor x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0: Tensor, t: Tensor, labels: Tensor, noise: Tensor = None) -> Tensor:
        """
        Training loss for the diffusion model.
        
        Args:
            x_0: Clean images tensor of shape (batch_size, channels, height, width)
            t: Timesteps tensor of shape (batch_size,)
            labels: Class labels tensor
            noise: Optional noise tensor (generated if None)
            
        Returns:
            MSE loss between predicted and actual noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Forward diffusion to get x_t
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.noise_predictor(x_t, t, labels)
        
        # Loss is MSE between actual and predicted noise
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, x_t: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        """
        Single step of the reverse diffusion sampling process.
        
        Args:
            x_t: Noisy image at timestep t
            t: Current timestep
            labels: Class labels tensor
            
        Returns:
            Sample from p(x_{t-1} | x_t)
        """
        # Predict noise
        predicted_noise = self.noise_predictor(x_t, t, labels)
        
        # Get posterior mean and variance
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Compute mean for posterior q(x_{t-1} | x_t, x_0)
        coef1 = torch.sqrt(1.0 / alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
        posterior_mean = coef1 * (x_t - coef2 * predicted_noise)
        
        # Add noise scaled by the posterior variance
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        variance = self.posterior_variance[t]
        
        return posterior_mean + torch.sqrt(variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, labels: Tensor) -> Tensor:
        """
        Full sampling loop to generate images from noise.
        
        Args:
            shape: Shape of the images to generate (batch_size, channels, height, width)
            labels: Class labels tensor
            
        Returns:
            Generated images
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=self.device)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch, labels)
        
        # Clamp values to valid image range
        return torch.clamp(x_t, -1., 1.)
    
    @torch.no_grad()
    def sample(self, batch_size: int, image_channels: int, image_size: int, labels: Tensor) -> Tensor:
        """
        Generate a batch of images from noise.
        
        Args:
            batch_size: Number of images to generate
            image_channels: Number of channels in the images
            image_size: Size of the images (assumed square)
            labels: Class labels tensor
            
        Returns:
            Generated images
        """
        return self.p_sample_loop(
            (batch_size, image_channels, image_size, image_size), 
            labels
        )
