"""
Simulate held-out market periods with a trained model (TimeGAN or the RGAN baseline) and audit the
generated order books.

The test hours are cut into --periods consecutive simulation periods. In each period, --windows
synthetic windows are generated, each starting from a real book of that period, and scored against
real windows of the same period: crossed books, unordered levels, KL of spread and mid-price returns,
SSIM of depth images, VWAP and volatility clustering. Real windows from the rest of the test hours
are scored the same way, as a reference for how much real periods differ from each other.

Outputs in --out-dir:
  period_summary.csv  one row per period with the audit's key numbers, for the README
  period_metrics.csv  every metric, per period, for real, synthetic and the real reference
  heatmaps.png        one real and one synthetic depth surface per period, from the same starting book
  book_period<p>.gif  with --gif: the same real and synthetic windows, animated step by step
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

from dataset import DATA_DIR, TICK, load_data, make_scaler, max_log_volume
from train import MODELS, generate_raw, real_windows
from utils import Evaluator, append_csv, describe, mid_price, print_table, spread_ticks, split_book, to_books

MESSAGE_PATH = DATA_DIR / "AMZN_2012-06-21_34200000_57600000_message_10.csv"


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Audit a trained model on consecutive periods of the test hours")
    p.add_argument("--checkpoint", type=Path, default=here / "outputs/e3b_gap_changes/last.pt")
    p.add_argument("--periods", type=int, default=5, help="number of consecutive simulation periods")
    p.add_argument("--windows", type=int, default=1_000, help="synthetic windows generated per period")
    p.add_argument("--depth-cents", type=int, default=35, help="price range around the mid in plots, in cents")
    p.add_argument("--gif", action="store_true", help="also animate each period's real and synthetic window")
    p.add_argument("--gif-fps", type=float, default=4, help="event steps per second in the GIFs")
    p.add_argument("--seed", type=int, default=3710)
    p.add_argument("--out-dir", type=Path, default=here / "outputs/predict")
    return p.parse_args()


def clock(seconds: float) -> str:
    """Seconds after midnight -> HH:MM:SS."""
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def depth_surface(w: np.ndarray, base: float, log_max: float, cents: int) -> np.ndarray:
    """One window [T, 40] as a [2 * cents + 1, T] price-by-time grid around price `base` (in cents).

    Cells hold log1p(volume) scaled to [0, 1]: positive for asks, negative for bids, 0 where no order rests.
    Levels outside the grid are not drawn.
    """
    ask_p, ask_v, bid_p, bid_v = split_book(w)
    grid = np.zeros((2 * cents + 1, len(w)))
    steps = np.broadcast_to(np.arange(len(w))[:, None], ask_p.shape)
    for prices, volumes, sign in ((ask_p, ask_v, 1), (bid_p, bid_v, -1)):
        rows = np.rint(prices / TICK - base).astype(int) + cents
        inside = (rows >= 0) & (rows <= 2 * cents)
        depth = np.log1p(np.clip(volumes, 0, None)) / log_max
        np.add.at(grid, (rows[inside], steps[inside]), sign * depth[inside])
    return grid


def centres(anchor: np.ndarray, synth: np.ndarray, anchored: bool) -> tuple:
    """Price (in cents) each panel is centred on: the shared starting book's mid-price, or for a model
    whose windows do not start from the real book (raw prices), the synthetic window's own first mid-price."""
    real_base = np.floor(mid_price(anchor) / TICK + 0.5)
    return real_base, real_base if anchored else np.floor(mid_price(synth[:1])[0] / TICK + 0.5)


def plot_heatmaps(pairs: list, log_max: float, cents: int, path: Path, anchored: bool = True) -> None:
    """One row per period: a real window and the synthetic window grown from the same starting book."""
    start = ("The synthetic window starts from the same real book." if anchored else
             "This model does not start from the real book, so its panels are centred on its own first mid-price.")
    fig = plt.figure(figsize=(11, 3.4 * len(pairs)), layout="constrained")
    fig.suptitle("Depth surfaces: asks red, bids blue, darker = more volume (log1p); black line = mid-price.\n"
                 f"{start} Levels beyond ±{cents} cents are not shown.", fontsize=10)
    for subfig, (title, anchor, real, synth) in zip(fig.subfigures(len(pairs), 1), pairs):
        subfig.suptitle(title, fontsize=10, fontweight="bold")
        panels = zip(subfig.subplots(1, 2), ("real", "synthetic"), (real, synth), centres(anchor, synth, anchored))
        for ax, label, w, base in panels:
            ax.imshow(depth_surface(w, base, log_max, cents), aspect="auto", origin="lower", cmap="RdBu_r",
                      vmin=-1, vmax=1, extent=(-0.5, len(w) - 0.5, -cents - 0.5, cents + 0.5))
            ax.plot(np.arange(len(w)), mid_price(w) / TICK - base, color="black", linewidth=1)
            spread = spread_ticks(w)
            moves = int((np.diff(mid_price(w)) != 0).sum())
            ax.set_title(f"{label}: spread {spread.min():.0f}-{spread.max():.0f} cents, mid moves {moves}/{len(w) - 1}, "
                         f"crossed {int((spread <= 0).sum())}", fontsize=8)
            ax.set(xlabel="event step", ylabel="cents from starting mid", ylim=(-cents - 0.5, cents + 0.5))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def animate_book(title: str, anchor: np.ndarray, real: np.ndarray, synth: np.ndarray, cents: int,
                 path: Path, fps: float, anchored: bool = True) -> None:
    """GIF of the book at each event step: volume at each price, real and synthetic side by side."""
    bases = [b * TICK / 1e4 for b in centres(anchor, synth, anchored)]  # in dollars
    _, ask_v, _, bid_v = split_book(np.stack([real, synth]))
    top = 1.1 * max(ask_v.max(), bid_v.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

    def draw(t):
        for ax, label, w, base in zip(axes, ("real", "synthetic"), (real, synth), bases):
            ax.cla()
            ask_p, ask_v, bid_p, bid_v = split_book(w[t])
            ask_p, bid_p = ask_p / 1e4, bid_p / 1e4  # dollars
            prices, volumes = np.concatenate([ask_p, bid_p]), np.concatenate([ask_v, bid_v])
            ax.bar(bid_p, bid_v, width=0.008, color="blue", alpha=0.85, label="Bids")
            ax.bar(ask_p, ask_v, width=0.008, color="red", alpha=0.7, label="Asks")
            ax.axvline(mid_price(w[t]) / 1e4, color="black", linestyle="--", label="Mid")
            ax.axvline((prices * volumes).sum() / volumes.sum(), color="grey", linestyle=":", label="Book VWAP")
            spread = spread_ticks(w[t])
            ax.set_title(f"{label}: step {t}/{len(w) - 1}, spread {spread:.0f} cents"
                         + (", CROSSED" if spread <= 0 else ""), fontsize=10)
            ax.set(xlim=(base - cents / 100, base + cents / 100), ylim=(0, top), xlabel="Price ($)")
            ax.legend(fontsize=8, loc="upper right")
        axes[0].set_ylabel("Volume")

    fig.suptitle(f"{title}: " + ("both windows start from the same real book" if anchored else
                                  "the synthetic window does not start from the real book"), fontsize=11)
    draw(0)
    fig.tight_layout()  # after drawing a frame, so the axis labels fit
    FuncAnimation(fig, draw, frames=len(real)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def summary(row: dict) -> dict:
    """The audit's key numbers for one period: synthetic against real, with the real reference."""
    real, synth, ref = row["real"], row["synthetic"], row["reference"]
    out = {"period": row["period"], "time": row["time"],
           "crossed_books": int(synth["crossed_count"]), "unordered_levels": synth["ladder"]}
    for key, name in (("kl_spread", "kl_spread"), ("kl_returns", "kl_returns"), ("ssim_avg_image", "ssim_avg_depth")):
        out[name], out[f"{name}_ref"] = synth[key], ref[key]
    for key, name in (("spread_mean_ticks", "mean_spread_cents"), ("vwap_ask_mean_usd", "vwap_ask_mean"),
                      ("vwap_ask_std_usd", "vwap_ask_std"), ("vwap_bid_mean_usd", "vwap_bid_mean"),
                      ("vwap_bid_std_usd", "vwap_bid_std"), ("acf_abs_r_lag1", "acf_abs_returns_lag1")):
        out[f"{name}_real"], out[f"{name}_synthetic"] = real[key], synth[key]
    out["mid_moves_real"], out["mid_moves_synthetic"] = 1 - real["zero_return_share"], 1 - synth["zero_return_share"]
    out["volume_variety"] = synth["volume_std_ratio"]
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("period_metrics.csv", "period_summary.csv"):
        (args.out_dir / name).unlink(missing_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generator = torch.Generator().manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")  # the scaler must stay on the CPU
    config = checkpoint["config"]
    representation, scaling = config["representation"], config.get("scaling", "minmax")
    seq_len = int(config["seq_len"])
    model_name = config.get("model", "timegan")
    print(f"checkpoint: {args.checkpoint} ({model_name}, {representation}, {scaling}), device: {device}")

    train_loader, _, test_loader, _ = load_data(seq_len, representation=representation, scaling=scaling)
    train_set, test_set = train_loader.dataset, test_loader.dataset
    scaler = make_scaler(scaling).load_state_dict(checkpoint["scaler"])
    model = MODELS[model_name](test_set.series.size(-1)).to(device)
    model.load_state_dict(checkpoint["model"])

    # The same depth colour scale and return bins as in training
    log_max = max_log_volume(train_set.features, representation)
    mid_ref = float(np.median(mid_price(real_windows(train_set, 4_000, representation, generator))))

    # Clock time of each test window's first and last snapshot. Message row i is the event that
    # produced order-book row i; feature row i is book row i + 1, except for raw features.
    times = np.loadtxt(MESSAGE_PATH, delimiter=",", usecols=0)
    first_row = len(times) - len(test_set.features)

    bounds = np.linspace(0, len(test_set), args.periods + 1).astype(int)
    summaries, pairs = [], []
    for p, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        span = f"{clock(times[first_row + lo])}-{clock(times[first_row + hi - 1])}"  # when its windows start
        start = torch.randint(lo, hi, (args.windows,), generator=generator)
        rest = torch.cat([torch.arange(0, lo), torch.arange(hi, len(test_set))])
        rest = rest[torch.randperm(len(rest), generator=generator)[:args.windows]]

        features, anchors = test_set.windows(start, scaled=False)
        real = to_books(features, anchors, representation)
        torch.manual_seed(args.seed + p)
        synth = generate_raw(model, scaler, anchors, seq_len, representation, device)  # same starting books
        reference = to_books(*test_set.windows(rest, scaled=False), representation)

        evaluate = Evaluator(real, log_max, mid_ref, seed=args.seed)
        row = {"period": p, "time": span, "real": describe(real), "synthetic": evaluate(synth),
               "reference": evaluate(reference)}
        summaries.append(summary(row))
        append_csv(args.out_dir / "period_summary.csv", summaries[-1])
        pairs.append((f"Period {p}, {span}", anchors[0].double().numpy(), real[0], synth[0]))
        print_table(f"Period {p} ({span}), {args.windows} windows",
                    {"real": row["real"], "synthetic": row["synthetic"], "real, rest of test (reference)": row["reference"]})
        for source in ("real", "synthetic", "reference"):
            # Real rows have no distance scores; leave those cells empty so every row has the same columns
            metrics = {k: row[source].get(k, "") for k in row["synthetic"]}
            append_csv(args.out_dir / "period_metrics.csv", {"period": p, "time": span, "source": source, **metrics})

    anchored = representation != "raw"  # raw windows do not start from the real book
    plot_heatmaps(pairs, log_max, args.depth_cents, args.out_dir / "heatmaps.png", anchored)
    if args.gif:
        for p, pair in enumerate(pairs, start=1):
            animate_book(*pair, args.depth_cents, args.out_dir / f"book_period{p}.gif", args.gif_fps, anchored)
    print_table("Summary (also in period_summary.csv)",
                {f"period {r['period']}": {k: v for k, v in r.items() if k not in ("period", "time")} for r in summaries})
    print(f"\noutputs in {args.out_dir}")


if __name__ == "__main__":
    main()
