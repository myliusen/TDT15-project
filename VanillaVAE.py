"""
Vanilla 1D VAE.
Heavily inspired by 
https://github.com/AntixK/PyTorch-VAE/blob/master/models/base.py and
https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py.

Key fixes vs. original via ChatGPT suggestions and rewriting:
- Updated decoder reshape & matching upsampling so output length == input length.
- GroupNorm instead of BatchNorm for stability with stochastic activations.
- Proper reconstruction loss: per-sample MSE sum (reduced by batch size) for meaningful gradients.
- Conv1d/ConvTranspose1d stacking with calculated flatten dimension via a dummy forward.
- Clear loss_function signature using keyword kld_weight (no positional confusion).
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Any, Tuple


class VanillaVAE1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        seq_length: int = 120,
        latent_dim: int = 16,
        hidden_dims: List[int] = None,
        gn_groups: int = 8,
    ) -> None:
        """A stable 1D VAE for time-series.

        Args:
            in_channels: number of input channels (1 for univariate series)
            seq_length: length of the input time series
            latent_dim: dimensionality of latent space
            hidden_dims: list of channel sizes for encoder e.g. [32,64,128]
            gn_groups: number of groups for GroupNorm
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.in_channels = in_channels
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.gn_groups = gn_groups

        # Build encoder: Conv1d layers that downsample by stride=2 each
        enc_modules = []
        in_ch = in_channels
        for h in hidden_dims:
            enc_modules.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, h, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(max(1, gn_groups), h),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            in_ch = h
        self.encoder = nn.Sequential(*enc_modules)

        # Compute flatten dimension dynamically with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_length)
            enc_out = self.encoder(dummy)
            self.enc_channels = enc_out.shape[1]  # e.g. 128
            self.enc_length = enc_out.shape[2]    # e.g. seq_length / 2^len(hidden_dims)
            self.flatten_dim = self.enc_channels * self.enc_length

        # Latent linear layers
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: linear to flatten_dim then ConvTranspose1d upsamples
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        # Build decoder transposed convolutions that mirror encoder
        # We'll upsample the same number of times the encoder downsampled
        dec_modules = []
        rev_hidden = hidden_dims[::-1]
        # We want to start at enc_channels and step towards smaller channels
        # Examples: rev_hidden = [128,64,32]
        for i in range(len(rev_hidden) - 1):
            in_ch = rev_hidden[i]
            out_ch = rev_hidden[i + 1]
            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.GroupNorm(max(1, gn_groups), out_ch),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

        # Final upsample layer: brings length back to original and reduces channels to last hidden dim
        # If the last step doesn't change channels, we still upsample once.
        # We intentionally do NOT add an extra upsampling beyond the number of encoder layers.
        dec_modules.append(
            nn.Sequential(
                # upsample from rev_hidden[-1] to rev_hidden[-1] if needed
                nn.ConvTranspose1d(
                    rev_hidden[-1],
                    rev_hidden[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.GroupNorm(max(1, gn_groups), rev_hidden[-1]),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        )

        self.decoder = nn.Sequential(*dec_modules)

        # Final conv to map to in_channels without changing length
        self.final_layer = nn.Sequential(
            nn.Conv1d(rev_hidden[-1], in_channels, kernel_size=3, padding=1)
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input into mu and logvar."""
        x_enc = self.encoder(x)
        x_flat = torch.flatten(x_enc, start_dim=1)
        mu = self.fc_mu(x_flat)
        log_var = self.fc_var(x_flat)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent z back to input space."""
        batch = z.size(0)
        result = self.decoder_input(z)  # (B, flatten_dim)
        # reshape to (B, enc_channels, enc_length)
        result = result.view(batch, self.enc_channels, self.enc_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        # ensure output has same length as input (may occur due to rounding) -> center-crop or pad
        if result.shape[2] != self.seq_length:
            # If slightly off due to conv arithmetic, adjust by cropping or padding
            diff = self.seq_length - result.shape[2]
            if diff > 0:
                # pad equally both sides
                left = diff // 2
                right = diff - left
                result = F.pad(result, (left, right))
            else:
                # crop
                left = (-diff) // 2
                result = result[:, :, left:left + self.seq_length]
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, x, mu, log_var

    def loss_function(self, recons: Tensor, input: Tensor, mu: Tensor, log_var: Tensor, kld_weight: float = 1.0) -> dict:
        """Compute VAE loss.

        - reconstruction loss is per-sample MSE summed across dimensions then averaged across batch.
        - KLD is the analytic KL divergence between N(mu, sigma^2) and N(0,1) averaged across batch.
        """
        # Reconstruction: sum per sample, then mean across batch -> comparable scale to KLD
        recons_loss = F.mse_loss(recons, input, reduction='sum') / input.size(0)

        # KLD: mean over batch
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kld_loss = torch.mean(kld_loss)

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach()
        }

    def sample(self, num_samples: int, device: torch.device = None) -> Tensor:
        device = device if device is not None else next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        recon, _, _, _ = self.forward(x)
        return recon