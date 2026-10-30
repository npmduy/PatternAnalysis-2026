"""
Data pipeline for the LOBSTER AMZN level-10 order book.

Loads the order book, splits it chronologically (70% train, 15% validation, 15% test), turns each
snapshot into model features (one of REPRESENTATIONS), scales them to [0, 1] (one of SCALINGS),
and serves overlapping windows of seq_len events. decode() turns generated features back into
order books, starting from the real book just before each window (its "anchor").

A raw book row has 40 columns: ask price, ask size, bid price, bid size for levels 1 to 10.
Prices are in dollars x 10,000.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent / "datasets/LOBSTER_SampleFile_AMZN_2012-06-21_10"
OB_PATH = DATA_DIR / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
TICK = 100.0  # one cent in LOBSTER price units (dollars x 10,000)
MAX_CENTS = 10_000  # cap on a rebuilt spread or gap ($100; the largest real one is $2.66)

# How an order book is presented to the model:
#   raw     - the 40 LOBSTER columns as they are
#   returns - log returns of the 20 prices and log1p of the 20 volumes, in the same column slots
#   gaps    - mid-price log return, log spread, log gaps between neighbouring levels (in cents)
#             and log1p of the 20 volumes; spreads and gaps are rebuilt as at least one cent, so
#             decoded books can never be crossed or have unordered levels
#   gap_changes - as gaps, but the log spread and log gaps are changes from the previous step,
#             so each window carries on from the spread and gaps of the real book it starts from
REPRESENTATIONS = ("raw", "returns", "gaps", "gap_changes")

# How features are squeezed into [0, 1] for the sigmoid outputs:
#   minmax   - linear, from the training minimum to the training maximum
#   quantile - by rank in the training data (empirical CDF)
SCALINGS = ("minmax", "quantile")


class MinMaxScaler:
    """Per-feature linear scaling to [0, 1], fitted on the training split."""

    def __init__(self, eps=1e-7):
        """eps avoids dividing by zero for a feature that never changes."""
        self._eps = eps
        self._min = None
        self._max = None

    def fit(self, x):
        """Learn each feature's minimum and maximum from x [rows, F]."""
        x = x.double()
        self._min = x.min(dim=0).values
        self._max = x.max(dim=0).values
        return self

    def transform(self, x):
        """Raw features [rows, F] -> scaled features in [0, 1] (float32)."""
        return ((x.double() - self._min) / (self._max - self._min + self._eps)).float()

    def fit_transform(self, x):
        """fit() then transform() on the same data."""
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        """Scaled features -> raw features, on the CPU in double precision (for evaluation)."""
        x = x.detach().cpu().double()
        return x * (self._max - self._min + self._eps) + self._min

    def inverse_transform_torch(self, x):
        """Differentiable inverse that stays on x's device and dtype (used inside the training loss)."""
        return x * (self._max - self._min + self._eps).to(x) + self._min.to(x)

    def state_dict(self):
        """What a checkpoint needs to rebuild the scaler."""
        return {"min": self._min, "max": self._max, "eps": self._eps}

    def load_state_dict(self, state):
        """Restore a scaler saved with state_dict()."""
        self._min = state["min"]
        self._max = state["max"]
        self._eps = state["eps"]
        return self


class QuantileScaler:
    """Per-feature empirical-CDF scaling to [0, 1], fitted on the training split.

    Each value maps to its rank in the training data. A value that repeats (such as the 85% of
    zero returns) owns its whole probability band: it maps to the middle of the band, and any
    model output inside the band decodes back to exactly that value.
    """

    def __init__(self, levels=2001):
        """levels is the number of stored quantiles per feature."""
        self._levels = levels
        self._q = None  # [levels, F]: training quantiles of each feature

    def fit(self, x):
        """Store the training quantiles of each feature of x [rows, F]."""
        self._q = torch.quantile(x.double(), torch.linspace(0, 1, self._levels, dtype=torch.float64), dim=0)
        return self

    def transform(self, x):
        """Raw features [rows, F] -> percentile in [0, 1], interpolated between stored quantiles."""
        x, k = x.double(), self._levels - 1
        out = torch.empty_like(x)
        for f in range(x.size(1)):
            q, v = self._q[:, f].contiguous(), x[:, f].contiguous()
            lo = torch.searchsorted(q, v)               # first quantile >= v
            hi = torch.searchsorted(q, v, right=True)   # first quantile > v
            i = lo.clamp(1, k)                          # between quantiles i-1 and i
            frac = ((v - q[i - 1]) / (q[i] - q[i - 1]).clamp_min(1e-300)).clamp(0, 1)
            u = (i - 1 + frac) / k
            on_grid = hi > lo                           # v equals one or more quantiles exactly
            u[on_grid] = (lo[on_grid] + hi[on_grid] - 1).double() / (2 * k)  # middle of its band
            out[:, f] = u
        return out.float()

    def fit_transform(self, x):
        """fit() then transform() on the same data."""
        return self.fit(x).transform(x)

    def _inverse(self, u):
        """Piecewise-linear inverse CDF; differentiable in u, flat over repeated values."""
        flat, k = u.reshape(-1, u.size(-1)), self._levels - 1
        q = self._q.to(flat)
        pos = flat.clamp(0, 1) * k
        # The bracketing quantile index is a constant; gradients flow through the weight w only
        i = pos.detach().floor().clamp(max=k - 1).long()
        w = pos - i
        x = q.gather(0, i) * (1 - w) + q.gather(0, i + 1) * w
        return x.reshape(u.shape)

    def inverse_transform(self, x):
        """Scaled features -> raw features, on the CPU in double precision (for evaluation)."""
        return self._inverse(x.detach().cpu().double())

    def inverse_transform_torch(self, x):
        """Differentiable inverse that stays on x's device and dtype (used inside the training loss)."""
        return self._inverse(x)

    def state_dict(self):
        """What a checkpoint needs to rebuild the scaler."""
        return {"quantiles": self._q, "levels": self._levels}

    def load_state_dict(self, state):
        """Restore a scaler saved with state_dict()."""
        self._q = state["quantiles"]
        self._levels = state["levels"]
        return self


def make_scaler(scaling="minmax"):
    """An unfitted scaler for the given scaling name."""
    if scaling == "minmax":
        return MinMaxScaler()
    if scaling == "quantile":
        return QuantileScaler()
    raise ValueError(f"scaling must be one of {SCALINGS}")


def book_structure(book):
    """Mid price [...] and log spread + 9 ask gaps + 9 bid gaps in cents [..., 19] of books [..., 40]."""
    ask_p, bid_p = book[..., 0::4], book[..., 2::4]  # every 4th column: price of each level
    mid = (ask_p[..., 0] + bid_p[..., 0]) / 2
    log_cents = torch.log(torch.cat([(ask_p[..., :1] - bid_p[..., :1]),
                                     torch.diff(ask_p, dim=-1), -torch.diff(bid_p, dim=-1)], dim=-1) / TICK)
    return mid, log_cents


def assemble_book(mid, log_cents, log_volumes):
    """Rebuild books [..., T, 40] from mid [..., T], log spread/gaps [..., T, 19] and log1p volumes [..., T, 20].

    Spread and gaps are rebuilt as at least one cent, so the book can never be crossed or unordered.
    They are also capped at MAX_CENTS, so prices stay small enough that a one-cent step is never
    lost to floating-point rounding.
    """
    cents = torch.exp(log_cents.clamp(max=math.log(MAX_CENTS))).clamp_min(1) * TICK
    half_spread, ask_gaps, bid_gaps = cents[..., 0] / 2, cents[..., 1:10], cents[..., 10:19]
    ask1, bid1 = (mid + half_spread)[..., None], (mid - half_spread)[..., None]
    # Each deeper level sits one gap further from the best price: a running sum of the gaps
    ask_p = torch.cat([ask1, ask1 + torch.cumsum(ask_gaps, dim=-1)], dim=-1)    # [..., T, 10]
    bid_p = torch.cat([bid1, bid1 - torch.cumsum(bid_gaps, dim=-1)], dim=-1)
    volumes = torch.expm1(log_volumes)
    book = torch.stack([ask_p, volumes[..., :10], bid_p, volumes[..., 10:]], dim=-1)
    return book.flatten(-2)  # LOBSTER order: ask price, ask size, bid price, bid size per level


def encode(book, representation="raw"):
    """Raw book [N, 40] -> (features, anchors), aligned row by row.

    anchors[i] is the raw book just before features[i]; decode() rebuilds prices from it.
    """
    book = book.double()
    if representation == "raw":
        return book, book
    if representation == "returns":
        features = torch.empty(len(book) - 1, book.size(1), dtype=torch.float64)
        features[:, 0::2] = torch.log(book[1:, 0::2] / book[:-1, 0::2])  # price log returns
        features[:, 1::2] = torch.log1p(book[1:, 1::2])                  # volumes
        return features, book[:-1]
    if representation in ("gaps", "gap_changes"):
        mid, log_cents = book_structure(book)
        structure = log_cents[1:] if representation == "gaps" else log_cents[1:] - log_cents[:-1]
        features = torch.cat([
            torch.log(mid[1:] / mid[:-1])[:, None],              # mid-price log return
            structure,                                           # log spread and 18 log gaps (or their changes)
            torch.log1p(torch.cat([book[1:, 1::4], book[1:, 3::4]], dim=1)),  # 20 volumes
        ], dim=1)
        return features, book[:-1]
    raise ValueError(f"representation must be one of {REPRESENTATIONS}")


def decode(features, anchor, representation="raw"):
    """Unscaled features [..., T, 40] -> raw book [..., T, 40], starting from anchor [..., 40].

    Differentiable, so the training loss can penalise invalid books.
    """
    if representation == "raw":
        return features
    if representation in ("gaps", "gap_changes"):
        mid0, log_cents0 = book_structure(anchor)
        # Price at step t = starting price x exp(sum of log returns up to t)
        mid = mid0[..., None] * torch.exp(torch.cumsum(features[..., 0], dim=-1))  # [..., T]
        log_cents = features[..., 1:20]
        if representation == "gap_changes":  # carry on from the starting book's spread and gaps
            log_cents = log_cents0[..., None, :] + torch.cumsum(log_cents, dim=-2)
        return assemble_book(mid, log_cents, features[..., 20:])
    # returns: each price column carries on from the same column of the anchor
    prices = anchor[..., None, 0::2] * torch.exp(torch.cumsum(features[..., 0::2], dim=-2))
    volumes = torch.expm1(features[..., 1::2])
    return torch.stack([prices, volumes], dim=-1).flatten(-2)  # back to price, volume, price, ...


def max_log_volume(features, representation="raw"):
    """log1p of the largest volume in unscaled features, which fixes the colour scale of depth images."""
    if representation in ("gaps", "gap_changes"):
        return features[:, 20:].max().item()  # volumes are the last 20 columns, already log1p
    top = features[:, 1::2].max().item()      # volumes are every other column
    return top if representation == "returns" else math.log1p(top)  # raw volumes still need log1p


class Windows(Dataset):
    """Overlapping windows over one split; each comes with the raw book just before its first step."""

    def __init__(self, x, features, anchors, seq_len=24):
        """x: scaled split [rows, F]; features: the same split unscaled; anchors: raw books [rows, 40]."""
        self._x = x                         # scaled: what the model sees
        self._features = features.float()   # unscaled: for evaluation that must not depend on the scaler
        self._anchors = anchors.float()
        self._seq_len = seq_len

    def __len__(self):
        """Number of windows: every row that has seq_len rows from it to the end of the split."""
        return len(self._x) - self._seq_len + 1

    def __getitem__(self, index):
        """The scaled window starting at row index [T, F] and its anchor [40]."""
        return self._x[index : index + self._seq_len], self._anchors[index]

    @property
    def series(self):
        """The whole scaled split [rows, F] the windows are cut from."""
        return self._x

    @property
    def features(self):
        """The whole split [rows, F] before scaling."""
        return self._features

    @property
    def seq_len(self):
        """Events per window."""
        return self._seq_len

    def windows(self, start, scaled=True):
        """Windows [n, T, F] beginning at rows start [n] (scaled, or unscaled features) and their anchors [n, F]."""
        source = self._x if scaled else self._features
        # start[:, None] + arange(T) is an [n, T] grid of row numbers, so one indexing call cuts every window
        return source[start[:, None] + torch.arange(self._seq_len)], self._anchors[start]

    def sample(self, n, generator=None, scaled=True):
        """n random windows [n, T, F] (scaled, or unscaled features) and their anchors [n, F]."""
        return self.windows(torch.randint(0, len(self), (n,), generator=generator), scaled)


def chrono_split(x, train_frac=0.7, val_frac=0.15):
    """Split rows in time order: first 70% train, next 15% validation, last 15% test (no shuffling)."""
    n = len(x)
    train_cut = int(train_frac * n)
    val_cut = int((train_frac + val_frac) * n)
    return x[:train_cut], x[train_cut:val_cut], x[val_cut:]


def load_data(seq_len=24, batch_size=128, representation="raw", scaling="minmax"):
    """Train, validation and test loaders of windows, plus the scaler fitted on the training split only.

    The training loader shuffles and drops the last incomplete batch; the others keep time order.
    """
    book = torch.from_numpy(np.loadtxt(OB_PATH, delimiter=",", dtype=np.int64))
    features, anchors = encode(book, representation)
    train, val, test = chrono_split(features)
    anchors_train, anchors_val, anchors_test = chrono_split(anchors)
    scaler = make_scaler(scaling).fit(train)  # fitted on training rows only, so no test information leaks in

    def loader(x, anchors, shuffle):
        """A DataLoader over the windows of one split."""
        return DataLoader(Windows(scaler.transform(x), x, anchors, seq_len), batch_size=batch_size,
                          shuffle=shuffle, drop_last=shuffle)

    return loader(train, anchors_train, True), loader(val, anchors_val, False), \
        loader(test, anchors_test, False), scaler


if __name__ == "__main__":
    # Quick check: one batch from every representation and scaling
    for representation in REPRESENTATIONS:
        for scaling in SCALINGS:
            train_loader, val_loader, test_loader, scaler = load_data(representation=representation, scaling=scaling)
            x, anchor = next(iter(train_loader))
            print(f"{representation}/{scaling}: batch {tuple(x.shape)} {x.dtype}, anchors {tuple(anchor.shape)}, "
                  f"range [{x.min():.3f}, {x.max():.3f}]")
