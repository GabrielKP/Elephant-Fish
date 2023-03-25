from typing import List, Union, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import locomotion
from model import FishSimple
from utils import to_device


@dataclass
class SimulationSimple:
    model: FishSimple
    n_bins_lin: int
    n_bins_ang: int
    n_bins_ori: int
    device: torch.device

    def run(self, n_steps: int, start_locomotion: Sequence) -> np.ndarray:
        """
        Parameters
        ----------
        n_steps: int
            amount of steps to simulate
        start_locomotion: List-like, len = 3
            First locomotion vector: [lin, ang, turn]

        Returns
        -------
        binned_locs_out: np.ndarray, shape = (n_steps, 3)
            binned locomotion output for all steps
        """
        start_locomotion = np.array(start_locomotion)

        # vec_CH_x = start_position[0] - start_position[2]
        # vec_CH_y = start_position[1] - start_position[3]
        # start_ori =

        prev_loc = torch.tensor(
            locomotion.bin_loc(
                start_locomotion[None, :],
                self.n_bins_lin,
                self.n_bins_ang,
                self.n_bins_ori,
            )[0]
        )

        self.model.eval()
        self.model.to(self.device)
        binned_locs_out = np.empty((n_steps, 3), dtype=np.int32)
        with torch.no_grad():
            for idx_step in range(n_steps):
                logit_loc_out = torch.stack(
                    self.model(*prev_loc.to(self.device)),
                    dim=0,
                )
                probs_loc_out = F.softmax(logit_loc_out, -1)
                binned_loc_out = torch.argmax(probs_loc_out, -1)
                binned_locs_out[idx_step] = binned_loc_out.cpu().numpy()
                prev_loc = binned_loc_out

        return binned_locs_out
