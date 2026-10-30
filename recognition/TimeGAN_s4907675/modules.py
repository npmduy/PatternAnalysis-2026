import torch
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        out, _ = self.gru(x)
        y = self.linear(out)
        return self.activation(y) if self.activation is not None else y

class Embedder(GRUNet):
    def __init__(self, in_dim, hidden_dim=24, num_layers=3):
        super().__init__(in_dim, hidden_dim, hidden_dim, num_layers, torch.sigmoid)
 
 
class Recovery(GRUNet):
    def __init__(self, out_dim, hidden_dim=24, num_layers=3):
        super().__init__(hidden_dim, hidden_dim, out_dim, num_layers, torch.sigmoid)
 
 
class Generator(GRUNet):
    def __init__(self, z_dim, hidden_dim=24, num_layers=3):
        super().__init__(z_dim, hidden_dim, hidden_dim, num_layers, torch.sigmoid)
 
 
class Supervisor(GRUNet):
    def __init__(self, hidden_dim=24, num_layers=3):
        super().__init__(hidden_dim, hidden_dim, hidden_dim, max(num_layers - 1, 1), torch.sigmoid)
 
 
class Discriminator(GRUNet):
    def __init__(self, hidden_dim=24, num_layers=3):
        super().__init__(hidden_dim, hidden_dim, 1, num_layers, None)

class TimeGAN(nn.Module):
    def __init__(self, d, z_dim=None, hidden_dim=24, num_layers=3):
        super().__init__()
        self.z_dim = z_dim or d
        self.embedder = Embedder(d, hidden_dim, num_layers)
        self.recovery = Recovery(d, hidden_dim, num_layers)
        self.generator = Generator(self.z_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = Discriminator(hidden_dim, num_layers)

    def sample_noise(self, n, seq_len, device=None):
        return torch.rand(n, seq_len, self.z_dim, device=device)

    def reconstruct(self, x):
        return self.recovery(self.embedder(x))

    def generate(self, n, seq_len, device=None):
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
        super().__init__()
        self.z_dim = z_dim or d
        self.generator = GRUNet(self.z_dim, hidden_dim, d, num_layers, torch.sigmoid)
        self.discriminator = GRUNet(d, hidden_dim, 1, num_layers, None)

    def sample_noise(self, n, seq_len, device=None):
        return torch.rand(n, seq_len, self.z_dim, device=device)

    def generate(self, n, seq_len, device=None):
        """A tuple whose first item is the generated windows, as for TimeGAN."""
        return (self.generator(self.sample_noise(n, seq_len, device)),)
