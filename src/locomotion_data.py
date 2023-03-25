import numpy as np
from torch.utils.data import Dataset


class LocDataset(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__()

        self.data = data

    def __getitem__(self, index) -> np.ndarray:
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]
