"""
Models: TimeGAN (Yoon et al., 2019) and the RGAN baseline.

Every network is a GRU followed by a linear layer. Shapes follow the README:
windows are [batch, time, features] and latent sequences are [batch, time, hidden].
"""

import torch
import torch.nn as nn


class GRUNet(nn.Module):
    """A multi-layer GRU followed by a linear layer applied at every time step."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation):
        """activation is applied to the output (e.g. torch.sigmoid), or None for raw logits."""
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        """[batch, time, in_dim] -> [batch, time, out_dim]."""
        out, _ = self.gru(x)
        y = self.linear(out)
        return self.activation(y) if self.activation is not None else y


class Embedder(GRUNet):
    """Maps a window in data space to a latent sequence (outputs in [0, 1])."""

    def __init__(self, in_dim, hidden_dim=24, num_layers=3):
        """in_dim is the number of features per event."""
        super().__init__(in_dim, hidden_dim, hidden_dim, num_layers, torch.sigmoid)


class Recovery(GRUNet):
    """Maps a latent sequence back to a window in data space (outputs in [0, 1], like the scaled data)."""

    def __init__(self, out_dim, hidden_dim=24, num_layers=3):
        """out_dim is the number of features per event."""
        super().__init__(hidden_dim, hidden_dim, out_dim, num_layers, torch.sigmoid)


class Generator(GRUNet):
    """Maps a noise sequence to a latent sequence."""

    def __init__(self, z_dim, hidden_dim=24, num_layers=3):
        """z_dim is the size of the noise vector at each time step."""
        super().__init__(z_dim, hidden_dim, hidden_dim, num_layers, torch.sigmoid)


class Supervisor(GRUNet):
    """Predicts the next step of a latent sequence from the steps so far."""

    def __init__(self, hidden_dim=24, num_layers=3):
        """Uses one GRU layer fewer than the other networks, as in the reference implementation."""
        super().__init__(hidden_dim, hidden_dim, hidden_dim, max(num_layers - 1, 1), torch.sigmoid)


class Discriminator(GRUNet):
    """Scores each step of a latent sequence: a logit for how real it looks."""

    def __init__(self, hidden_dim=24, num_layers=3):
        """Outputs one raw logit per time step; the loss applies the sigmoid."""
        super().__init__(hidden_dim, hidden_dim, 1, num_layers, None)


class TimeGAN(nn.Module):
    """The five TimeGAN networks. The training steps that use them are in train.py."""

    def __init__(self, d, z_dim=None, hidden_dim=24, num_layers=3):
        """d is the number of features per event; the noise has the same size unless z_dim is given."""
        super().__init__()
        self.z_dim = z_dim or d
        self.embedder = Embedder(d, hidden_dim, num_layers)
        self.recovery = Recovery(d, hidden_dim, num_layers)
        self.generator = Generator(self.z_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = Discriminator(hidden_dim, num_layers)

    def sample_noise(self, n, seq_len, device=None):
        """Uniform noise in [0, 1], shaped [n, seq_len, z_dim]."""
        return torch.rand(n, seq_len, self.z_dim, device=device)

    def reconstruct(self, x):
        """Embed real windows and recover them: x -> h -> x_tilde."""
        return self.recovery(self.embedder(x))

    def generate(self, n, seq_len, device=None):
        """Generate n windows: noise -> e_hat (generator) -> h_hat (supervisor) -> x_hat (recovery).

        Returns (x_hat, h_hat, e_hat); the latent sequences are needed for the training losses.
        """
        z = self.sample_noise(n, seq_len, device)
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        x_hat = self.recovery(h_hat)
        return x_hat, h_hat, e_hat


class RGAN(nn.Module):
    """Baseline recurrent GAN: RCGAN (Esteban et al., 2017) without a condition, like our TimeGAN.

    A GRU generator maps noise straight to data, and a GRU discriminator scores each step of a real
    or generated window. There is no embedding, supervisor or moment loss: only the adversarial loss.
    """

    def __init__(self, d, z_dim=None, hidden_dim=24, num_layers=3):
        """d is the number of features per event; the noise has the same size unless z_dim is given."""
        super().__init__()
        self.z_dim = z_dim or d
        self.generator = GRUNet(self.z_dim, hidden_dim, d, num_layers, torch.sigmoid)
        self.discriminator = GRUNet(d, hidden_dim, 1, num_layers, None)

    def sample_noise(self, n, seq_len, device=None):
        """Uniform noise in [0, 1], shaped [n, seq_len, z_dim]."""
        return torch.rand(n, seq_len, self.z_dim, device=device)

    def generate(self, n, seq_len, device=None):
        """A tuple whose first item is the generated windows, as for TimeGAN."""
        return (self.generator(self.sample_noise(n, seq_len, device)),)
