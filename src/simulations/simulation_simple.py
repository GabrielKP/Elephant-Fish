from dataclasses import dataclass
from random import choices
from typing import Dict, List, Union, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.locomotion import bin_loc, unbin_loc, convLocToCart
from src.models.fish_simple import FishSimple


@dataclass
class SimulationSimple:
    """
    model: SimpleFish
        Model with which to do simulation.
    n_bins_lin: int
    n_bins_ang: int
    n_bins_ori: int
        amount of bins for each movement type
    device: torch.device
        torch device on which to place/run the model
    """

    model: FishSimple
    n_bins_lin: int
    n_bins_ang: int
    n_bins_ori: int
    device: torch.device

    def run(
        self,
        n_steps: int,
        start_locomotion: Sequence,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        n_steps: int
            amount of steps to simulate
        start_locomotion: List-like, len = 3
            First locomotion vector: [lin, ang, turn]
        sample: bool, default=True
            Whether to sample from output. If set to False, will
            use argmax for next locomotion.
        temperature: float, default=1.0,
            Higher temperature makes distribution more uniform,
            lower temperature places more emphasis on few high
            predictions.

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
            bin_loc(
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
                logit_lin, logit_ang, logit_ori = self.model(
                    *prev_loc.to(self.device)
                )

                probs_lin = F.softmax(logit_lin / temperature, -1)
                probs_ang = F.softmax(logit_ang / temperature, -1)
                probs_ori = F.softmax(logit_ori / temperature, -1)

                if sample:
                    binned_lin = choices(
                        list(range(self.n_bins_lin)), probs_lin, k=1
                    )[0]
                    binned_ang = choices(
                        list(range(self.n_bins_ang)), probs_ang, k=1
                    )[0]
                    binned_ori = choices(
                        list(range(self.n_bins_ori)), probs_ori, k=1
                    )[0]
                else:
                    binned_lin = torch.argmax(probs_lin, -1).cpu().item()
                    binned_ang = torch.argmax(probs_ang, -1).cpu().item()
                    binned_ori = torch.argmax(probs_ori, -1).cpu().item()

                binned_locs_out[idx_step] = (
                    binned_lin,
                    binned_ang,
                    binned_ori,
                )
                prev_loc = torch.tensor((binned_lin, binned_ang, binned_ori))

        return binned_locs_out


def run_simulation(
    config: Dict[str, Union[str, float, int]],
    model: FishSimple,
    device: torch.device,
    start_positions: Sequence,
    start_locomotion: Sequence,
) -> np.ndarray:
    start_positions = np.array(start_positions)
    start_locomotion = np.array(start_locomotion)

    # get sim
    sim = SimulationSimple(
        model=model,
        n_bins_lin=model.n_bins_lin,
        n_bins_ang=model.n_bins_ang,
        n_bins_ori=model.n_bins_ori,
        device=device,
    )

    # run
    binned_loc = sim.run(
        config["n_steps"],
        start_locomotion,
        sample=config["sample"],
        temperature=config["temperature"],
    )

    # unbin
    unbinned_loc = unbin_loc(
        binned_locomotion=binned_loc,
        n_bins_lin=model.n_bins_lin,
        n_bins_ang=model.n_bins_ang,
        n_bins_ori=model.n_bins_ori,
    )

    # to cartesian
    tracks = convLocToCart(
        unbinned_loc,
        start_positions,
    )

    return tracks
