from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__()

        self.data = data

    def __getitem__(self, index) -> np.ndarray:
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]


class WallRayDataset(Dataset):
    def __init__(self, data_loc, data_wr, label_loc) -> None:
        super().__init__()

        self.data_loc = data_loc
        self.data_wr = data_wr
        self.label_loc = label_loc

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.data_loc[index],
            self.data_wr[index],
            self.label_loc[index],
        )

    def __len__(self) -> int:
        return self.data_loc.shape[0]
