import logging
from typing import Any, Dict, Union, Sequence

import numpy as np
import torch

FORMAT = "[%(levelname)s] %(name)s.%(funcName)s - %(message)s"

logging.basicConfig(format=FORMAT)


def get_logger(
    name=__name__,
    log_level=logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter(FORMAT)

    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


log = get_logger(__name__)


def get_device(config) -> torch.device:
    """Returns device."""
    if config["device"] is not None:
        if config["device"] == "cuda" and not torch.cuda.is_available():
            log.critical(f"Device {config['device']} not available.")
            raise Exception(f"Device {config['device']} not available.")

        return torch.device(config["device"])
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(
    items: Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]],
    device: torch.device,
) -> Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]]:
    # it is a list/tuple!
    if isinstance(items, Sequence):
        return [element.to(device) for element in items]
    # it is a dict!
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in items.items()
    }


def is_sequence(item: Any) -> bool:
    return isinstance(item, Sequence) or isinstance(item, np.ndarray)


def to_tensor(
    items: Union[
        Sequence[Union[np.ndarray, Any]], Dict[str, Union[np.ndarray, Any]]
    ],
    targets: bool = False,
) -> Union[Sequence[torch.Tensor], Dict[str, Union[torch.Tensor, Any]]]:
    dtype = torch.long if targets else torch.int
    # tuple/list
    if isinstance(items, Sequence):
        return [
            torch.tensor(element, dtype=dtype)
            if is_sequence(element)
            else element
            for element in items
        ]
    # dict
    return {
        key: torch.tensor(value, dtype=dtype) if is_sequence(value) else value
        for key, value in items.items()
    }
