"""
Train vanilla TimeGAN (Yoon et al., 2019) on LOBSTER AMZN level-10 order books.

Three phases: embedder and recovery pretraining, supervisor training, and joint
adversarial training. Phase 3 is validated every --eval-every iterations against real
validation windows; the best checkpoint is then scored once on the held-out test hours.

--representation picks how books are fed to the model: raw columns, price log returns with
log1p volumes, or gaps (mid-price return, spread and gaps between levels, which makes crossed
books and unordered levels impossible). --scaling picks how features are squeezed into [0, 1]
(min-max, or by percentile among the training values). --lambda-valid adds a penalty on
generated books that are crossed or have unordered price levels (0 = vanilla).

Outputs in --out-dir: training_curves.png, val_metrics.csv, last.pt, best.pt,
results.json, test_distributions.png and test_depth.png.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import REPRESENTATIONS, SCALINGS, decode, load_data, max_log_volume
from modules import TimeGAN
from utils import (TICK, Evaluator, append_csv, describe, mid_price, plot_depth, plot_distributions,
                   plot_training, print_table, to_books, to_raw)

D_UPDATE_THRESHOLD = 0.15  # the discriminator only trains while its loss is above this


def parse_args():
    p = argparse.ArgumentParser(description="Train vanilla TimeGAN on LOBSTER AMZN level-10 order books")
    p.add_argument("--iters", type=int, default=10_000, help="iterations per training phase")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--representation", choices=REPRESENTATIONS, default="raw")
    p.add_argument("--scaling", choices=SCALINGS, default="minmax")
    p.add_argument("--lambda-valid", type=float, default=0.0,
                   help="weight of the penalty on crossed books and unordered price levels")
    p.add_argument("--eval-every", type=int, default=1_000, help="phase-3 iterations between validations")
    p.add_argument("--eval-windows", type=int, default=1_000, help="windows generated per validation")
    p.add_argument("--test-windows", type=int, default=4_000, help="windows generated for the final test")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=3710)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return p.parse_args()


# ----------------------------------------------------------------------------- losses and steps

def bce_logits(logits, target_logit) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target_logit))

def moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    std_hat = torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
    loss = (std_hat - std).abs().mean() + (x_hat.mean(dim=0) - x.mean(dim=0)).abs().mean()
    return loss

def validity_penalty(book: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Hinge penalty, in ticks, on books whose best bid is not at least `margin` ticks below the
    best ask, or whose price levels are less than `margin` ticks apart. Zero for valid books."""
    ask_p, bid_p = book[..., 0::4] / TICK, book[..., 2::4] / TICK
    crossed = F.relu(bid_p[..., 0] - ask_p[..., 0] + margin)
    ask_ladder = F.relu(ask_p[..., :-1] - ask_p[..., 1:] + margin).sum(-1)
    bid_ladder = F.relu(bid_p[..., 1:] - bid_p[..., :-1] + margin).sum(-1)
    return (crossed + ask_ladder + bid_ladder).mean()

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

def generator_step(model: TimeGAN, optim: torch.optim.Optimizer, x: torch.Tensor, anchor: torch.Tensor,
                   gamma: float, lambda_valid: float, to_book) -> dict:
    x_hat, h_hat, e_hat = model.generate(x.size(0), x.size(1), x.device)
    h = model.embedder(x)

    loss_g_adv = bce_logits(model.discriminator(h_hat), 1.0)
    loss_g_adv_e = bce_logits(model.discriminator(e_hat), 1.0)
    loss_s = F.mse_loss(model.supervisor(h)[:, :-1], h[:, 1:])
    loss_v = moment_loss(x_hat, x)
    loss_valid = validity_penalty(to_book(x_hat, anchor))  # generated windows start from the real anchors
    loss = loss_g_adv + gamma * loss_g_adv_e + 100 * torch.sqrt(loss_s) + 100 * loss_v + lambda_valid * loss_valid

    model.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    terms = torch.stack([loss, loss_g_adv, loss_g_adv_e, loss_s, loss_v, loss_valid]).detach().tolist()  # one GPU sync
    return dict(zip(("g_total", "g_adv", "g_adv_e", "g_sup", "g_moment", "g_valid"), terms))

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
    if loss.item() > D_UPDATE_THRESHOLD:
        model.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    return loss.item()


# ----------------------------------------------------------------------------- helpers

def forever(loader: torch.utils.data.DataLoader):
    while True:
        yield from loader

def adam(*nets: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam([p for net in nets for p in net.parameters()], lr=lr)

@torch.no_grad()
def generate_raw(model: TimeGAN, scaler, anchors: torch.Tensor, seq_len: int, representation: str,
                 device) -> np.ndarray:
    """One synthetic window per anchor (a real starting book), in raw LOBSTER units."""
    model.eval()
    x_hat, _, _ = model.generate(len(anchors), seq_len, device)
    model.train()
    return to_raw(scaler, x_hat.cpu(), anchors, representation)

def real_windows(dataset, n: int, representation: str, generator: torch.Generator) -> np.ndarray:
    """n random real windows from one split, in raw LOBSTER units (built without the scaler)."""
    features, anchors = dataset.sample(n, generator, scaled=False)
    return to_books(features, anchors, representation)

@torch.no_grad()
def recon_mse(model: TimeGAN, x: torch.Tensor, device) -> float:
    model.eval()
    x = x.to(device)
    mse = F.mse_loss(model.reconstruct(x), x).item()
    model.train()
    return mse

def save_checkpoint(path: Path, model: TimeGAN, scaler, history: dict, args) -> None:
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    torch.save({"model": model.state_dict(), "scaler": scaler.state_dict(),
                "history": dict(history), "config": config}, path)

def log_progress(phase: str, i: int, args, t0: float, history: dict, keys: list) -> None:
    """Every --log-every iterations: mean of the recent losses, speed and time left in the phase."""
    if (i + 1) % args.log_every and i + 1 != args.iters:
        return
    ms = (time.perf_counter() - t0) / (i + 1) * 1000
    losses = "  ".join(f"{k} {np.mean(history[k][-args.log_every:]):.4g}" for k in keys)
    print(f"[{phase}] {i + 1}/{args.iters}  {losses}  ({ms:.1f} ms/it, {ms * (args.iters - i - 1) / 60000:.1f} min left)")


# ----------------------------------------------------------------------------- main

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "val_metrics.csv").unlink(missing_ok=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)  # for sampling real evaluation windows
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    print(f"representation: {args.representation}, scaling: {args.scaling}, lambda_valid: {args.lambda_valid}")

    train_loader, val_loader, test_loader, scaler = load_data(args.seq_len, args.batch_size, args.representation,
                                                              args.scaling)
    train_set, val_set, test_set = train_loader.dataset, val_loader.dataset, test_loader.dataset

    # Fixed real windows that generated windows are scored against
    real = {
        "train": real_windows(train_set, args.test_windows, args.representation, generator),
        "val": real_windows(val_set, args.eval_windows, args.representation, generator),
        "test": real_windows(test_set, args.test_windows, args.representation, generator),
    }
    log_max = max_log_volume(train_set.features, args.representation)
    mid_ref = float(np.median(mid_price(real["train"])))
    val_eval = Evaluator(real["val"], log_max, mid_ref, seed=args.seed)
    test_eval = Evaluator(real["test"], log_max, mid_ref, seed=args.seed)
    val_reference = val_eval(real["train"])  # what a perfect copy of the training hours would score
    val_recon_x = val_set.sample(2_000, generator)[0]

    def to_book(x_hat, anchor):
        """Model output -> raw book, differentiable, for the validity penalty."""
        return decode(scaler.inverse_transform_torch(x_hat), anchor, args.representation)

    model = TimeGAN(train_set.series.size(-1)).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    batches = forever(train_loader)
    history = defaultdict(list)  # loss name -> one value per iteration of its phase
    timing = {}

    # Phase 1: embedder and recovery
    t0 = time.perf_counter()
    optim = adam(model.embedder, model.recovery, lr=args.lr)
    for i in range(args.iters):
        x, _ = next(batches)
        history["recon"].append(reconstruction_step(model, optim, x.to(device)))
        log_progress("phase 1", i, args, t0, history, ["recon"])
    timing["phase1_s"] = time.perf_counter() - t0
    print(f"phase 1 done: validation reconstruction MSE {recon_mse(model, val_recon_x, device):.5f}")

    # Phase 2: supervisor (the reference code optimises G and S here; only S receives gradients)
    t0 = time.perf_counter()
    optim = adam(model.generator, model.supervisor, lr=args.lr)
    for i in range(args.iters):
        x, _ = next(batches)
        history["supervised"].append(supervisor_step(model, optim, x.to(device)))
        log_progress("phase 2", i, args, t0, history, ["supervised"])
    timing["phase2_s"] = time.perf_counter() - t0

    # Phase 3: joint training, two generator and embedder updates per discriminator update
    t0 = time.perf_counter()
    optim_g = adam(model.generator, model.supervisor, lr=args.lr)
    optim_e = adam(model.embedder, model.recovery, lr=args.lr)
    optim_d = adam(model.discriminator, lr=args.lr)
    val_rows, best = [], {"score": float("inf"), "iter": None}
    d_updates, last_eval, eval_s = 0, 0, 0.0
    for i in range(args.iters):
        g_terms, e_losses = [], []
        for _ in range(2):
            x, anchor = (t.to(device) for t in next(batches))
            g_terms.append(generator_step(model, optim_g, x, anchor, args.gamma, args.lambda_valid, to_book))
            e_losses.append(embedder_step(model, optim_e, x))
        for k in g_terms[0]:
            history[k].append(float(np.mean([g[k] for g in g_terms])))
        history["e_recon"].append(float(np.mean(e_losses)))
        x, _ = next(batches)
        loss_d = discriminator_step(model, optim_d, x.to(device), args.gamma)
        history["d"].append(loss_d)
        d_updates += loss_d > D_UPDATE_THRESHOLD
        log_progress("phase 3", i, args, t0, history, ["g_total", "g_valid", "d", "e_recon"])

        if (i + 1) % args.eval_every == 0 or i + 1 == args.iters:
            t_eval = time.perf_counter()
            anchors = val_set.sample(args.eval_windows, generator)[1]
            synth = generate_raw(model, scaler, anchors, args.seq_len, args.representation, device)
            row = {"iter": i + 1, "d_update_share": d_updates / (i + 1 - last_eval),
                   "recon_val": recon_mse(model, val_recon_x, device), **val_eval(synth)}
            d_updates, last_eval = 0, i + 1
            val_rows.append(row)
            append_csv(args.out_dir / "val_metrics.csv", row)
            save_checkpoint(args.out_dir / "last.pt", model, scaler, history, args)
            if row["kl_spread"] + row["kl_returns"] < best["score"]:
                best = {"score": row["kl_spread"] + row["kl_returns"], "iter": i + 1}
                save_checkpoint(args.out_dir / "best.pt", model, scaler, history, args)
            plot_training(history, val_rows, val_reference, args.out_dir / "training_curves.png")
            eval_s += time.perf_counter() - t_eval
            print(f"[validate] {i + 1}/{args.iters}  KL spread {row['kl_spread']:.3f}  KL returns {row['kl_returns']:.3f}  "
                  f"SSIM {row['ssim_pairs']:.3f} pairs / {row['ssim_avg_image']:.3f} avg  crossed {row['crossed']:.3f}  "
                  f"any violation {row['any_violation']:.3f}  D updates {row['d_update_share']:.2f}")
    timing["phase3_s"] = time.perf_counter() - t0 - eval_s
    timing["validation_s"] = eval_s

    # Test once on the held-out final hours: last and best checkpoints vs the real-data reference
    test_anchors = test_set.sample(args.test_windows, generator)[1]  # generated windows start from real test books
    synth_last = generate_raw(model, scaler, test_anchors, args.seq_len, args.representation, device)
    model.load_state_dict(torch.load(args.out_dir / "best.pt", map_location=device)["model"])
    synth_best = generate_raw(model, scaler, test_anchors, args.seq_len, args.representation, device)
    results = {
        "real test": describe(real["test"]),
        "real train (reference)": test_eval(real["train"]),
        f"TimeGAN best (it {best['iter']})": test_eval(synth_best),
        "TimeGAN last": test_eval(synth_last),
    }
    print_table("Test: held-out final hours", results)
    plot_distributions(real["test"], real["train"], synth_best, test_eval.return_unit,
                       args.out_dir / "test_distributions.png")
    plot_depth(real["test"], synth_best, log_max, args.out_dir / "test_depth.png")

    for phase in ("phase1", "phase2", "phase3"):
        timing[f"{phase}_ms_per_iter"] = 1000 * timing[f"{phase}_s"] / args.iters
    if device.type == "cuda":
        timing["gpu"] = torch.cuda.get_device_name()
        timing["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    summary = {"best_iter": best["iter"], "timing": timing, "results": results,
               "config": {k: str(v) for k, v in vars(args).items()}}
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntraining time {sum(timing[f'phase{p}_s'] for p in (1, 2, 3)) / 60:.1f} min "
          f"(+ {eval_s / 60:.1f} min validation); outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
