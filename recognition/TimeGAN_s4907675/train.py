import torch
import torch.nn.functional as F

from dataset import load_data
from modules import TimeGAN

N_ITERS = 10
LR = 1e-3
SEED = 3710

def bce_logits(logits, target_logit) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target_logit))

def moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    std_hat = torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
    loss = (std_hat - std).abs().mean() + (x_hat.mean(dim=0) - x.mean(dim=0)).abs().mean()
    return loss

def reconstruction_step(model: TimeGAN, optim: torch.optim.Optimizer, x: torch.Tensor) -> float:
    mse = F.mse_loss(model.reconstruct(x), x)
    loss = 10 * torch.sqrt(mse)
    model.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return mse.item()

def supervisor_step(model: TimeGAN, optim: torch.optim.Optimizer, x: torch.Tensor) -> float:
    h = model.embedder(x)
    loss = F.mse_loss(model.supervisor(h)[:, :-1], h[:, 1:])
    model.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return loss.item()

def generator_step(model: TimeGAN, optim: torch.optim.Optimizer, x: torch.Tensor, gamma: float) -> float:
    x_hat, h_hat, e_hat = model.generate(x.size(0), x.size(1), x.device)
    h = model.embedder(x)

    loss_g_adv = bce_logits(model.discriminator(h_hat), 1.0)
    loss_g_adv_e = bce_logits(model.discriminator(e_hat), 1.0)
    loss_s = F.mse_loss(model.supervisor(h)[:, :-1], h[:, 1:])
    loss_v = moment_loss(x_hat, x)
    loss = loss_g_adv + gamma * loss_g_adv_e + 100 * torch.sqrt(loss_s) + 100 * loss_v

    model.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    return loss.item()

def embedder_step(model: TimeGAN, optim: torch.optim.Optimizer, x: torch.Tensor) -> float:
    h = model.embedder(x)
    mse = F.mse_loss(model.recovery(h), x)
    loss = 10 * torch.sqrt(mse) + 0.1 * F.mse_loss(model.supervisor(h)[:, :-1], h[:, 1:])
    model.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return mse.item()

def discriminator_step(model: TimeGAN, optim: torch.optim.Optimizer, x: torch.Tensor, gamma) -> float:
    with torch.no_grad():
        h = model.embedder(x)
        _, h_hat, e_hat = model.generate(x.size(0), x.size(1), x.device)

    loss = bce_logits(model.discriminator(h), 1.0) + bce_logits(model.discriminator(h_hat), 0.0) + gamma * bce_logits(model.discriminator(e_hat), 0.0)
    if loss.item() > 0.15:
        model.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    return loss.item()

def forever(loader: torch.utils.data.DataLoader):
    while True:
        yield from loader

if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_loader, val_loader, test_loader, scaler = load_data()
    model = TimeGAN(40).to(device)
    batches = forever(train_loader)

    optim = torch.optim.Adam([p for p in model.embedder.parameters()] + [p for p in model.recovery.parameters()], lr=LR)
    for i in range(N_ITERS):
        batch = next(batches).to(device)
        loss = reconstruction_step(model, optim, batch)

    optim = torch.optim.Adam([p for p in model.generator.parameters()] + [p for p in model.supervisor.parameters()], lr=LR)
    for i in range(N_ITERS):
        batch = next(batches).to(device)
        loss = supervisor_step(model, optim, batch)

    optim_g = torch.optim.Adam([p for p in model.generator.parameters()] + [p for p in model.supervisor.parameters()], lr=LR)
    optim_e = torch.optim.Adam([p for p in model.embedder.parameters()] + [p for p in model.recovery.parameters()], lr=LR)
    optim_d = torch.optim.Adam([p for p in model.discriminator.parameters()], lr=LR)
    for i in range(N_ITERS):
        for _ in range(2):
            batch = next(batches).to(device)
            loss_g = generator_step(model, optim_g, batch, 1.0)
            loss_e = embedder_step(model, optim_e, batch)
        batch = next(batches).to(device)
        loss_d = discriminator_step(model, optim_d, batch, 1.0)

    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict()}, "outputs/test.pt")