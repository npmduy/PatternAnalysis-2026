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

def max_log_volume(scaler, representation="raw"):
    """log1p of the largest training volume, which fixes the colour scale of depth images."""
    top = scaler.state_dict()["max"][1::2].max().item()
    return top if representation == "returns" else math.log1p(top)

class Windows(Dataset):
    """Overlapping windows over one split; each comes with the raw book just before its first step."""

    def __init__(self, x, anchors, seq_len=24):
        self._x = x
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
    def seq_len(self):
        return self._seq_len

    def sample(self, n, generator=None):
        """n random windows [n, T, F] and their anchors [n, F]."""
        start = torch.randint(0, len(self), (n,), generator=generator)
        return self._x[start[:, None] + torch.arange(self._seq_len)], self._anchors[start]

def chrono_split(x, train_frac=0.7, val_frac=0.15):
    n = len(x)
    train_cut = int(train_frac * n)
    val_cut = int((train_frac + val_frac) * n)
    return x[:train_cut], x[train_cut:val_cut], x[val_cut:]

def load_data(seq_len=24, batch_size=128, representation="raw"):
    book = torch.from_numpy(np.loadtxt(OB_PATH, delimiter=",", dtype=np.int64))
    features, anchors = encode(book, representation)
    train, val, test = chrono_split(features)
    anchors_train, anchors_val, anchors_test = chrono_split(anchors)
    scaler = MinMaxScaler().fit(train)

    def loader(x, anchors, shuffle):
        return DataLoader(Windows(scaler.transform(x), anchors, seq_len), batch_size=batch_size,
                          shuffle=shuffle, drop_last=shuffle)

    return loader(train, anchors_train, True), loader(val, anchors_val, False), \
        loader(test, anchors_test, False), scaler

if __name__ == "__main__":
    for representation in REPRESENTATIONS:
        train_loader, val_loader, test_loader, scaler = load_data(representation=representation)
        x, anchor = next(iter(train_loader))
        print(f"{representation}: batch {tuple(x.shape)} {x.dtype}, anchors {tuple(anchor.shape)}, "
              f"range [{x.min():.3f}, {x.max():.3f}]")
