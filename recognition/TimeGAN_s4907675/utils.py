"""
Evaluation metrics and plots for generated LOBSTER level-10 order books.

Metrics work on windows in raw LOBSTER units, shaped [n, T, 40], with columns
ask price, ask size, bid price, bid size for levels 1..10. Prices are dollars x 10,000.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset import TICK, decode

TARGET_KL = 0.1
TARGET_SSIM = 0.6
ACF_LAGS = 10
SPREAD_EDGES = np.arange(-10.5, 81.5)  # 1-tick bins centred on -10..80 ticks


# ----------------------------------------------------------------------------- windows

def to_books(features: torch.Tensor, anchors: torch.Tensor, representation: str = "raw") -> np.ndarray:
    """Unscaled windows [n, T, F] -> raw books, with prices on the tick grid and whole-share volumes.

    anchors [n, F] are the real books each window starts from (used by the returns representation).
    """
    raw = decode(features.detach().cpu().double(), anchors.detach().cpu().double(), representation).numpy()
    # Round halves up rather than to even (np.round): with round-half-to-even, two levels one cent
    # apart that both sit on a half cent (101.5, 102.5) would round to the same price.
    raw[..., 0::2] = np.floor(raw[..., 0::2] / TICK + 0.5) * TICK
    raw[..., 1::2] = np.round(raw[..., 1::2])
    return raw


def to_raw(scaler, x: torch.Tensor, anchors: torch.Tensor, representation: str = "raw") -> np.ndarray:
    """Scaled model windows [n, T, F] -> raw books (see to_books)."""
    n, t, f = x.shape
    return to_books(scaler.inverse_transform(x.reshape(-1, f)).reshape(n, t, f), anchors, representation)


# ----------------------------------------------------------------------------- book quantities

def split_book(w: np.ndarray):
    """Ask price, ask volume, bid price, bid volume, each [..., 10]."""
    return w[..., 0::4], w[..., 1::4], w[..., 2::4], w[..., 3::4]


def spread_ticks(w: np.ndarray) -> np.ndarray:
    ask_p, _, bid_p, _ = split_book(w)
    return (ask_p[..., 0] - bid_p[..., 0]) / TICK


def mid_price(w: np.ndarray) -> np.ndarray:
    ask_p, _, bid_p, _ = split_book(w)
    return (ask_p[..., 0] + bid_p[..., 0]) / 2


def vwap(w: np.ndarray):
    """Volume-weighted average price of each side over the 10 levels: (ask, bid), each [n, T]."""
    ask_p, ask_v, bid_p, bid_v = split_book(w)
    ask = (ask_p * ask_v).sum(-1) / np.maximum(ask_v.sum(-1), 1e-9)
    bid = (bid_p * bid_v).sum(-1) / np.maximum(bid_v.sum(-1), 1e-9)
    return ask, bid


def log_returns(price: np.ndarray) -> np.ndarray:
    """Log returns along time inside each window, [n, T] -> [n, T-1]; never across windows."""
    price = np.maximum(price, 1e-9)
    return np.log(price[:, 1:] / price[:, :-1])


# ----------------------------------------------------------------------------- metrics

def violations(w: np.ndarray) -> dict:
    """Order-book rules broken per snapshot: crossed or locked book, unordered price ladder, negative volume."""
    ask_p, ask_v, bid_p, bid_v = split_book(w)
    crossed = bid_p[..., 0] >= ask_p[..., 0]
    ladder = (np.diff(ask_p, axis=-1) <= 0).any(-1) | (np.diff(bid_p, axis=-1) >= 0).any(-1)
    negative = (ask_v < 0).any(-1) | (bid_v < 0).any(-1)
    return {
        "crossed": crossed.mean(),
        "crossed_count": crossed.sum(),
        "ladder": ladder.mean(),
        "neg_volume": negative.mean(),
        "any_violation": (crossed | ladder | negative).mean(),
    }


def kl_divergence(real: np.ndarray, other: np.ndarray, edges: np.ndarray, eps: float = 1e-6) -> float:
    """KL(real || other) between histograms on fixed bins; values beyond the edges go to the end bins."""
    real = np.clip(np.ravel(real), edges[0], edges[-1])
    other = np.clip(np.ravel(other), edges[0], edges[-1])
    p = np.histogram(real, edges)[0] / real.size + eps
    q = np.histogram(other, edges)[0] / other.size + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def acf_abs(r: np.ndarray, lags: int = ACF_LAGS) -> np.ndarray:
    """Autocorrelation of |returns| at lags 1..lags, pooled over windows (volatility clustering)."""
    a = np.abs(r) - np.abs(r).mean()
    var = (a ** 2).mean()
    if var == 0:
        return np.zeros(lags)
    return np.array([(a[:, :-k] * a[:, k:]).mean() / var for k in range(1, lags + 1)])


def depth_images(w: np.ndarray, log_max: float) -> np.ndarray:
    """[n, 20, T] images of log1p(volume): rows ask10..ask1 then bid1..bid10, scaled to [0, 1]."""
    _, ask_v, _, bid_v = split_book(w)
    img = np.concatenate([ask_v[..., ::-1], bid_v], axis=-1)
    img = np.log1p(np.clip(img, 0, None)) / log_max
    return np.clip(img, 0, 1).transpose(0, 2, 1)


def describe(w: np.ndarray) -> dict:
    """Statistics of one set of windows, real or synthetic."""
    mid = mid_price(w)
    r = log_returns(mid)
    ask_vwap, bid_vwap = vwap(w)
    std = r.std()
    stats = {
        **violations(w),
        "spread_mean_ticks": spread_ticks(w).mean(),
        "mid_mean_usd": mid.mean() / 1e4,
        "vwap_ask_mean_usd": ask_vwap.mean() / 1e4,
        "vwap_ask_std_usd": ask_vwap.std() / 1e4,
        "vwap_bid_mean_usd": bid_vwap.mean() / 1e4,
        "vwap_bid_std_usd": bid_vwap.std() / 1e4,
        "zero_return_share": (r == 0).mean(),
        "return_std_bp": std * 1e4,
        "excess_kurtosis": ((r - r.mean()) ** 4).mean() / std ** 4 - 3 if std > 0 else np.nan,
        "acf_abs_r_lag1": acf_abs(r)[0],
        "acf_abs_vwap_r_lag1": acf_abs(log_returns((ask_vwap + bid_vwap) / 2))[0],
    }
    return {k: float(v) for k, v in stats.items()}


class Evaluator:
    """Scores sets of windows against one fixed sample of real windows, all in raw units."""

    def __init__(self, real: np.ndarray, log_max: float, mid_ref: float, seed: int = 0, ssim_pairs: int = 300):
        self.log_max = log_max
        self.return_unit = float(np.log((mid_ref + TICK / 2) / mid_ref))  # log return of a half-tick mid move
        self.return_edges = (np.arange(-40, 42) - 0.5) * self.return_unit  # bins centred on 0, ±1 .. ±40 half-ticks
        self.ssim_pairs = ssim_pairs
        self.rng = np.random.default_rng(seed)
        self.spread = spread_ticks(real)
        self.returns = log_returns(mid_price(real))
        self.images = depth_images(real, log_max)
        self.acf = acf_abs(self.returns)
        self.volume_std = real[..., 1::2].reshape(-1, real.shape[-1] // 2).std(0)

    def __call__(self, w: np.ndarray) -> dict:
        """All metrics for windows w: their own statistics plus distances to the real sample."""
        r = log_returns(mid_price(w))
        images = depth_images(w, self.log_max)
        i = self.rng.integers(0, len(self.images), self.ssim_pairs)
        j = self.rng.integers(0, len(images), self.ssim_pairs)
        volume_std = w[..., 1::2].reshape(-1, w.shape[-1] // 2).std(0)
        scores = {
            "kl_spread": kl_divergence(self.spread, spread_ticks(w), SPREAD_EDGES),
            "kl_returns": kl_divergence(self.returns, r, self.return_edges),
            "ssim_pairs": np.mean([structural_similarity(self.images[a], images[b], data_range=1.0)
                                   for a, b in zip(i, j)]),
            "ssim_avg_image": structural_similarity(self.images.mean(0), images.mean(0), data_range=1.0),
            "acf_abs_r_gap": np.abs(acf_abs(r) - self.acf).mean(),
            "volume_std_ratio": (volume_std / np.maximum(self.volume_std, 1e-9)).mean(),
        }
        return {**describe(w), **{k: float(v) for k, v in scores.items()}}


# ----------------------------------------------------------------------------- logging

def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def print_table(title: str, columns: dict) -> None:
    """Print metrics side by side: one column per set of windows, one row per metric."""
    names = list(columns)
    keys = list(dict.fromkeys(k for col in columns.values() for k in col))
    width = max(22, max(len(k) for k in keys) + 2)
    print(f"\n{title}")
    print(f"{'metric':{width}s}" + "".join(f"{n:>24s}" for n in names))
    for k in keys:
        cells = "".join(f"{columns[n][k]:>24.4g}" if k in columns[n] else f"{'-':>24s}" for n in names)
        print(f"{k:{width}s}{cells}")


# ----------------------------------------------------------------------------- plots

def _smooth(values, window: int = 100):
    """Moving average for noisy loss curves: (x, y)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 * window:
        return np.arange(len(values)), values
    return np.arange(window - 1, len(values)), np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training(history: dict, val_rows: list, reference: dict, path: Path) -> None:
    """Loss curves of every phase, and validation metrics against the targets and real-data reference."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    def losses(ax, title, keys, log=False):
        for k in keys:
            if history.get(k):
                raw, = ax.plot(history[k], linewidth=0.4, alpha=0.3)
                ax.plot(*_smooth(history[k]), linewidth=1.2, color=raw.get_color(), label=k)
        ax.set(title=title, xlabel="iteration in phase", yscale="log" if log else "linear")

    def validation(ax, title, keys, target=None, log=False, show_reference=False):
        for k in keys:
            line, = ax.plot([r["iter"] for r in val_rows], [r[k] for r in val_rows], marker="o", ms=3, label=k)
            if show_reference:
                ax.axhline(reference[k], color=line.get_color(), linestyle=":", linewidth=1,
                           label=f"{k}: real train vs val")
        if target is not None:
            ax.axhline(target, color="red", linestyle="--", linewidth=1, label="target")
        ax.set(title=title, xlabel="phase-3 iteration", yscale="log" if log else "linear")

    losses(axes[0, 0], "Phase 1: reconstruction MSE", ["recon"], log=True)
    losses(axes[0, 1], "Phase 2: supervised loss", ["supervised"], log=True)
    losses(axes[0, 2], "Phase 3: adversarial losses", ["g_adv", "g_adv_e", "d"])
    axes[0, 2].axhline(3 * np.log(2), color="grey", linestyle=":", linewidth=1, label="D guessing (3 ln 2)")
    losses(axes[1, 0], "Phase 3: generator supervised, moment and validity terms",
           ["g_sup", "g_moment", "g_valid"], log=True)
    losses(axes[1, 1], "Phase 3: embedder reconstruction MSE", ["e_recon"], log=True)
    validation(axes[1, 2], "Validation: reconstruction MSE", ["recon_val"], log=True)
    validation(axes[2, 0], "Validation: KL(real || synthetic)", ["kl_spread", "kl_returns"], TARGET_KL,
               log=True, show_reference=True)
    validation(axes[2, 1], "Validation: SSIM of depth images", ["ssim_pairs", "ssim_avg_image"], TARGET_SSIM,
               show_reference=True)
    validation(axes[2, 2], "Validation: share of snapshots breaking a rule", ["crossed", "ladder", "any_violation"])
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_experiments(runs: dict, path: Path) -> None:
    """Compare training runs: the loss of each phase and the validation metrics, one line per run.

    runs maps a label to an output folder written by train.py (last.pt and val_metrics.csv).
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for label, folder in runs.items():
        folder = Path(folder)
        history = torch.load(folder / "last.pt", map_location="cpu")["history"]
        with (folder / "val_metrics.csv").open() as f:
            val = list(csv.DictReader(f))
        for ax, key in zip(axes[0], ("recon", "supervised", "d")):
            ax.plot(*_smooth(history[key], 200), linewidth=1.2, label=label)
        iters = [int(row["iter"]) for row in val]
        for ax, key in zip(axes[1], ("kl_returns", "kl_spread", "any_violation")):
            ax.plot(iters, [float(row[key]) for row in val], marker="o", ms=3, label=label)

    axes[0, 0].set(title="Phase 1: reconstruction MSE", yscale="log")
    axes[0, 1].set(title="Phase 2: supervised loss", yscale="log")
    axes[0, 2].set(title="Phase 3: discriminator loss")
    axes[0, 2].axhline(3 * np.log(2), color="grey", linestyle=":", linewidth=1, label="guessing (3 ln 2)")
    axes[1, 0].set(title="Validation: KL of mid-price returns", yscale="log")
    axes[1, 1].set(title="Validation: KL of spread", yscale="log")
    axes[1, 2].set(title="Validation: share of invalid snapshots")
    for ax in axes[1, :2]:
        ax.axhline(TARGET_KL, color="red", linestyle="--", linewidth=1, label="target 0.1")
    for ax in axes[0]:
        ax.set_xlabel("iteration in phase (moving average of 200)")
    for ax in axes[1]:
        ax.set_xlabel("phase-3 iteration")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_distributions(real: np.ndarray, train: np.ndarray, synth: np.ndarray, return_unit: float, path: Path) -> None:
    """Spread, mid-price moves, volatility clustering and VWAP: real test vs real train vs synthetic."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for w, label in ((real, "real test"), (train, "real train"), (synth, "synthetic")):
        mid_moves = log_returns(mid_price(w)) / return_unit
        ask_vwap, bid_vwap = vwap(w)
        axes[0, 0].hist(np.clip(spread_ticks(w).ravel(), -10, 80), SPREAD_EDGES, density=True,
                        histtype="step", label=label)
        axes[0, 1].hist(np.clip(mid_moves.ravel(), -20, 20), np.arange(-20.5, 21.5), density=True,
                        histtype="step", label=label)
        axes[1, 0].plot(range(1, ACF_LAGS + 1), acf_abs(log_returns(mid_price(w))), marker="o", label=label)
        axes[1, 1].hist(((ask_vwap + bid_vwap) / 2e4).ravel(), 60, density=True, histtype="step", label=label)
    axes[0, 0].set(title="Spread (ticks, clipped to [-10, 80])", xlabel="ticks")
    axes[0, 1].set(title="Mid-price move per step (half-ticks, clipped)", xlabel="half-ticks", yscale="log")
    axes[1, 0].set(title="ACF of |mid return| (volatility clustering)", xlabel="lag")
    axes[1, 1].set(title="VWAP (average of ask and bid side)", xlabel="USD", yscale="log")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_depth(real: np.ndarray, synth: np.ndarray, log_max: float, path: Path, n: int = 4, seed: int = 0) -> None:
    """Depth heatmaps (levels x time) of random real and synthetic windows on the same colour scale."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.5), squeeze=False)
    for row, (label, w) in enumerate((("real test", real), ("synthetic", synth))):
        idx = rng.choice(len(w), n, replace=False)
        for col, (k, img) in enumerate(zip(idx, depth_images(w[idx], log_max))):
            ax = axes[row, col]
            ax.imshow(img, aspect="auto", cmap="magma", vmin=0, vmax=1)
            ax.axhline(9.5, color="white", linewidth=0.8)
            crossed = int((spread_ticks(w[k]) <= 0).sum())
            ax.set_title(f"{label}: spread {spread_ticks(w[k]).mean():.1f} ticks, crossed {crossed}", fontsize=8)
            ax.set_yticks([0, 9, 10, 19], ["ask10", "ask1", "bid1", "bid10"], fontsize=7)
            ax.set_xlabel("step")
    fig.suptitle("Depth as log1p(volume), same colour scale; the white line separates asks (top) from bids")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
