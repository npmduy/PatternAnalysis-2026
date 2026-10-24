from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent / "datasets/LOBSTER_SampleFile_AMZN_2012-06-21_10"
OB_PATH = Path(DATA_DIR) / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

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

    def state_dict(self):
        return {"min": self._min, "max": self._max, "eps": self._eps}

    def load_state_dict(self, state):
        self._min = state["min"]
        self._max = state["max"]
        self._eps = state["eps"]
        return self

class Windows(Dataset):
    def __init__(self, x, seq_len=24):
        self._x = x
        self._seq_len = seq_len

    def __len__(self):
        return len(self._x) - self._seq_len + 1

    def __getitem__(self, index):
        return self._x[index : index + self._seq_len]

    @property
    def series(self):
        """The whole scaled split [rows, F] the windows are cut from."""
        return self._x

def chrono_split(x, train_frac=0.7, val_frac=0.15):
    n = len(x)
    train_cut = int(train_frac * n)
    val_cut = int((train_frac + val_frac) * n)
    return x[:train_cut], x[train_cut:val_cut], x[val_cut:]

def load_data(seq_len=24, batch_size=128):
    book = torch.from_numpy(np.loadtxt(OB_PATH, delimiter=",", dtype=np.int64))
    train, val, test = chrono_split(book)
    scaler = MinMaxScaler().fit(train)
    train = scaler.transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    return DataLoader(Windows(train, seq_len), batch_size=batch_size, shuffle=True, drop_last=True), \
        DataLoader(Windows(val, seq_len), batch_size=batch_size, shuffle=False), \
        DataLoader(Windows(test, seq_len), batch_size=batch_size, shuffle=False), \
        scaler

if __name__ == "__main__":
    train_loader, val_loader, test_loader, scaler = load_data()
    x = next(iter(train_loader))
    print("batch:", tuple(x.shape), x.dtype, f"range [{x.min():.3f}, {x.max():.3f}]")