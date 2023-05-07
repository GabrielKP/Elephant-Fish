from typing import Tuple

import numpy as np
import pandas as pd

from src.locomotion import getnLoc
from src.reader import extract_coordinates


def load_raycasts(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def load_locomotions(path: str = "data/sleap/diff1.h5") -> np.ndarray:
    tracks = extract_coordinates(
        path,
        [b"head", b"center"],
    )
    return getnLoc(tracks, 1)
