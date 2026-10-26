import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent / "datasets/LOBSTER_SampleFile_AMZN_2012-06-21_10"
OB_PATH = Path(DATA_DIR) / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

# How an order book is presented to the model:
#   raw     - the 40 LOBSTER columns as they are
#   returns - log returns of the 20 prices and log1p of the 20 volumes, in the same column slots
REPRESENTATIONS = ("raw", "returns")

# How features are squeezed into [0, 1] for the sigmoid outputs:
#   minmax   - linear, from the training minimum to the training maximum
#   quantile - by rank in the training data (empirical CDF)
SCALINGS = ("minmax", "quantile")

class MinMaxScaler:
    def __init__(self, eps=1e-7):
        self._eps = eps
        self._min = None
        self._max = None

    def fit(self, x):
        x = x.double()
        self._min = x.min(dim=0).values
        self._max = x.max(dim=0).values
        return self

    def transform(self, x):
        return ((x.double() - self._min) / (self._max - self._min + self._eps)).float()

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        x = x.detach().cpu().double()
        return x * (self._max - self._min + self._eps) + self._min

    def inverse_transform_torch(self, x):
        """Differentiable inverse that stays on x's device and dtype (used inside the training loss)."""
        return x * (self._max - self._min + self._eps).to(x) + self._min.to(x)

    def state_dict(self):
        return {"min": self._min, "max": self._max, "eps": self._eps}

    def load_state_dict(self, state):
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
        self._levels = levels
        self._q = None  # [levels, F]: training quantiles of each feature

    def fit(self, x):
        self._q = torch.quantile(x.double(), torch.linspace(0, 1, self._levels, dtype=torch.float64), dim=0)
        return self

    def transform(self, x):
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
        return self.fit(x).transform(x)

    def _inverse(self, u):
        """Piecewise-linear inverse CDF; differentiable in u, flat over repeated values."""
        flat, k = u.reshape(-1, u.size(-1)), self._levels - 1
        q = self._q.to(flat)
        pos = flat.clamp(0, 1) * k
        i = pos.detach().floor().clamp(max=k - 1).long()
        w = pos - i
        x = q.gather(0, i) * (1 - w) + q.gather(0, i + 1) * w
        return x.reshape(u.shape)

    def inverse_transform(self, x):
        return self._inverse(x.detach().cpu().double())

    def inverse_transform_torch(self, x):
        """Differentiable inverse that stays on x's device and dtype (used inside the training loss)."""
        return self._inverse(x)

    def state_dict(self):
        return {"quantiles": self._q, "levels": self._levels}

    def load_state_dict(self, state):
        self._q = state["quantiles"]
        self._levels = state["levels"]
        return self

def make_scaler(scaling="minmax"):
    if scaling == "minmax":
        return MinMaxScaler()
    if scaling == "quantile":
        return QuantileScaler()
    raise ValueError(f"scaling must be one of {SCALINGS}")

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
    raise ValueError(f"representation must be one of {REPRESENTATIONS}")

def decode(features, anchor, representation="raw"):
    """Unscaled features [..., T, 40] -> raw book [..., T, 40], starting from anchor [..., 40].

    Differentiable, so the training loss can penalise invalid books.
    """
    if representation == "raw":
        return features
    prices = anchor[..., None, 0::2] * torch.exp(torch.cumsum(features[..., 0::2], dim=-2))
    volumes = torch.expm1(features[..., 1::2])
    return torch.stack([prices, volumes], dim=-1).flatten(-2)  # back to price, volume, price, ...

def max_log_volume(features, representation="raw"):
    """log1p of the largest volume in unscaled features, which fixes the colour scale of depth images."""
    top = features[:, 1::2].max().item()
    return top if representation == "returns" else math.log1p(top)

class Windows(Dataset):
    """Overlapping windows over one split; each comes with the raw book just before its first step."""

    def __init__(self, x, features, anchors, seq_len=24):
        self._x = x                         # scaled: what the model sees
        self._features = features.float()   # unscaled: for evaluation that must not depend on the scaler
        self._anchors = anchors.float()
        self._seq_len = seq_len

    def __len__(self):
        return len(self._x) - self._seq_len + 1

    def __getitem__(self, index):
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
        return self._seq_len

    def sample(self, n, generator=None, scaled=True):
        """n random windows [n, T, F] (scaled, or unscaled features) and their anchors [n, F]."""
        start = torch.randint(0, len(self), (n,), generator=generator)
        source = self._x if scaled else self._features
        return source[start[:, None] + torch.arange(self._seq_len)], self._anchors[start]

def chrono_split(x, train_frac=0.7, val_frac=0.15):
    n = len(x)
    train_cut = int(train_frac * n)
    val_cut = int((train_frac + val_frac) * n)
    return x[:train_cut], x[train_cut:val_cut], x[val_cut:]

def load_data(seq_len=24, batch_size=128, representation="raw", scaling="minmax"):
    book = torch.from_numpy(np.loadtxt(OB_PATH, delimiter=",", dtype=np.int64))
    features, anchors = encode(book, representation)
    train, val, test = chrono_split(features)
    anchors_train, anchors_val, anchors_test = chrono_split(anchors)
    scaler = make_scaler(scaling).fit(train)

    def loader(x, anchors, shuffle):
        return DataLoader(Windows(scaler.transform(x), x, anchors, seq_len), batch_size=batch_size,
                          shuffle=shuffle, drop_last=shuffle)

    return loader(train, anchors_train, True), loader(val, anchors_val, False), \
        loader(test, anchors_test, False), scaler

if __name__ == "__main__":
    for representation in REPRESENTATIONS:
        for scaling in SCALINGS:
            train_loader, val_loader, test_loader, scaler = load_data(representation=representation, scaling=scaling)
            x, anchor = next(iter(train_loader))
            print(f"{representation}/{scaling}: batch {tuple(x.shape)} {x.dtype}, anchors {tuple(anchor.shape)}, "
                  f"range [{x.min():.3f}, {x.max():.3f}]")
