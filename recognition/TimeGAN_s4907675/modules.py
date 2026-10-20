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