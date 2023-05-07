import os
import json
from typing import Any, Dict

import torch
import torch.nn as nn

from src.utils import check_make_dirs, get_logger

log = get_logger(__name__)


class Fish(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, X):
        return NotImplementedError("Forward not implemented")

    def save(self, config: Dict[str, Any]) -> None:
        dir = config["model_dir"]
        check_make_dirs(dir, verbose=False)
        path_statedict = os.path.join(dir, "statedict.pt")
        torch.save(self.state_dict(), path_statedict)
        log.info(f"Saved statedict to {path_statedict}")

        path_config = os.path.join(dir, "config.json")
        with open(path_config, "w") as f_out:
            json.dump(config, f_out, indent=4)
        log.info(f"Saved config to to {path_config}")

    def load(self, config: Dict[str, Any]) -> Dict[str, Any]:
        dir = config["model_dir"]
        path_statedict = os.path.join(dir, "statedict.pt")
        self.load_state_dict(torch.load(path_statedict))

        path_config = os.path.join(dir, "config.json")
        with open(path_config, "r") as f_in:
            saved_config = json.load(f_in)
        return saved_config

    @staticmethod
    def _from_dir(model_class, dir: str, **override_kwargs) -> "Fish":
        """Load config and initialize Fish from it.

        Parameters
        ----------
        dir : str
            path to model dir
        **override kwargs
            all additional kwargs are used to update the
            loaded config
        """
        # load and update config
        path_config = os.path.join(dir, "config.json")
        with open(path_config, "r") as f_in:
            saved_config: Dict = json.load(f_in)
        saved_config.update(override_kwargs)

        # init model
        model = model_class(saved_config)

        # init savedict
        model.load(saved_config)

        return model
