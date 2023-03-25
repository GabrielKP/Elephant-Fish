import numpy as np
import pandas as pd

import locomotion
import reader


def load_raycasts(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def load_locomotions(path: str) -> np.ndarray:
    tracks = reader.extract_coordinates(
        "data/sleap/diff1.h5",
        [b"head", b"center"],
    )
    return locomotion.getnLoc(tracks, 1)
