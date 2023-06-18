import os
import json
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.raycasts import get_egocentric_wall_ray_orientations
from src.utils import check_make_dirs, get_logger
from src.models.fish import Fish

log = get_logger(__name__)


class FishWallRays(Fish):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.n_bins_lin = config["n_bins_lin"]
        self.n_bins_ang = config["n_bins_ang"]
        self.n_bins_ori = config["n_bins_ori"]
        self.n_wall_rays = config["n_wall_rays"]
        self.n_bins_wall_rays = config["n_bins_wall_rays"]
        self.field_of_view = config["field_of_view"]
        self.egocentric_wall_ray_orientations = (
            get_egocentric_wall_ray_orientations(
                self.n_wall_rays, self.field_of_view
            )
        )
        self.max_view = config["max_view"]

        self.embed_dim_lin = config["embed_dim_lin"]
        self.embed_dim_ang = config["embed_dim_ang"]
        self.embed_dim_ori = config["embed_dim_ori"]
        self.embed_dim_wall_rays = config["embed_dim_wall_rays"]

        self.hidden_dim = config["hidden"]

        # init layers
        self.embed_lin = nn.Embedding(self.n_bins_lin, self.embed_dim_lin)
        self.embed_ang = nn.Embedding(self.n_bins_ang, self.embed_dim_ang)
        self.embed_ori = nn.Embedding(self.n_bins_ori, self.embed_dim_ori)

        # add wall_ray layers
        for idx_wr in range(self.n_wall_rays):
            self.add_module(
                f"embed_wall_ray_{idx_wr:03}",
                nn.Embedding(self.n_bins_wall_rays, self.embed_dim_wall_rays),
            )

        self.hidden = nn.Linear(
            self.embed_dim_lin
            + self.embed_dim_ang
            + self.embed_dim_ori
            + (self.n_wall_rays * self.embed_dim_wall_rays),
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
        ids_wall_ray: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed_lin = self.embed_lin(id_lin)
        embed_ang = self.embed_ang(id_ang)
        embed_ori = self.embed_ori(id_ori)

        embeds_ls = [embed_lin, embed_ang, embed_ori]
        for idx_wr, id_wall_ray in enumerate(range(ids_wall_ray.shape[1])):
            embeds_ls.append(
                self.get_submodule(f"embed_wall_ray_{idx_wr:03}")(
                    ids_wall_ray[:, id_wall_ray]
                )
            )

        embeds = torch.cat(embeds_ls, -1)

        hidden = self.do(self.af(self.hidden(embeds)))

        out_lin = self.fc_out_lin(hidden)
        out_ang = self.fc_out_ang(hidden)
        out_ori = self.fc_out_ori(hidden)

        return (out_lin, out_ang, out_ori)

    @staticmethod
    def from_dir(dir: str, **override_kwargs) -> "FishWallRays":
        return Fish._from_dir(FishWallRays, dir, **override_kwargs)
