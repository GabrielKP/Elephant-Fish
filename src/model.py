import os
import json
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import check_make_dirs, get_logger

log = get_logger(__name__)


class FishSimple(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.n_bins_lin = config["n_bins_lin"]
        self.n_bins_ang = config["n_bins_ang"]
        self.n_bins_ori = config["n_bins_ori"]

        self.embed_dim_lin = config["embed_dim_lin"]
        self.embed_dim_ang = config["embed_dim_ang"]
        self.embed_dim_ori = config["embed_dim_ori"]

        self.hidden_dim = config["hidden"]

        # init layers
        self.embed_lin = nn.Embedding(self.n_bins_lin, self.embed_dim_lin)
        self.embed_ang = nn.Embedding(self.n_bins_ang, self.embed_dim_ang)
        self.embed_ori = nn.Embedding(self.n_bins_ori, self.embed_dim_ori)

        self.hidden = nn.Linear(
            self.embed_dim_lin + self.embed_dim_ang + self.embed_dim_ori,
            self.hidden_dim,
        )
        self.af = nn.ReLU()
        self.do = nn.Dropout(config["dropout"])

        self.fc_out_lin = nn.Linear(self.hidden_dim, self.n_bins_lin)
        self.fc_out_ang = nn.Linear(self.hidden_dim, self.n_bins_ang)
        self.fc_out_ori = nn.Linear(self.hidden_dim, self.n_bins_ori)

    def forward(
        self,
        id_lin: torch.Tensor,
        id_ang: torch.Tensor,
        id_ori: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed_lin = self.embed_lin(id_lin)
        embed_ang = self.embed_ang(id_ang)
        embed_ori = self.embed_ori(id_ori)

        embeds = torch.cat((embed_lin, embed_ang, embed_ori), -1)

        hidden = self.do(self.af(self.hidden(embeds)))

        out_lin = self.fc_out_lin(hidden)
        out_ang = self.fc_out_ang(hidden)
        out_ori = self.fc_out_ori(hidden)

        return (out_lin, out_ang, out_ori)

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
    def from_dir(dir: str, **override_kwargs) -> "FishSimple":
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
        model = FishSimple(saved_config)

        # init savedict
        model.load(saved_config)

        return model
