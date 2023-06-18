from dataclasses import dataclass
from random import choices
from typing import Dict, List, Union, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from src.functions import convPolarToCart, getAngle, getDistance

from src.locomotion import bin_loc, unbin_loc, convLocToCart, row_l2c
from src.models.fish_wall_rays import FishWallRays
from src.raycasts import bin_wall_rays, raycasts_from_tracks


@dataclass
class SimulationWallRays:
    """
    model: FishWallRays
        Model with which to do simulation.
    n_bins_lin: int
    n_bins_ang: int
    n_bins_ori: int
        amount of bins for each movement type
    device: torch.device
        torch device on which to place/run the model
    """

    model: FishWallRays
    n_bins_lin: int
    n_bins_ang: int
    n_bins_ori: int
    n_bins_wall_rays: int
    n_wall_rays: int
    field_of_view: Tuple[float, float]
    egocentric_wall_ray_orientations: np.ndarray
    max_view: float
    device: torch.device

    def _get_binned_wall_ray(self, oriposition: np.ndarray) -> torch.Tensor:
        # correct input format
        tracks = np.empty((1, 4))
        tracks[0, 2:4] = oriposition[0:2]
        orientations = oriposition[2:3]
        # get raycasts
        _, wall_distances, wall_intersections = raycasts_from_tracks(
            tracks,
            self.egocentric_wall_ray_orientations,
            orientations=orientations,
        )
        # wall_distances.shape = (1, n_wall_rays)
        # wall_intersections.shape = (1, n_wall_rays, 2)
        # bin
        binned_wall_distances = torch.tensor(
            bin_wall_rays(
                wall_distances,
                self.n_bins_wall_rays,
                (0.0, self.max_view),
            )[0]
        )  # shape = (n_wall_rays)
        return binned_wall_distances, wall_distances[0], wall_intersections[0]

    def run(
        self,
        n_steps: int,
        start_locomotion: Sequence,
        start_position: np.ndarray,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        tracks: np.ndarray: shape = (n_steps + 1, 4)
            trackset for fish center
        wall_distances: np.ndarray, shape = (n_steps, n_wall_rays)
            distance of wall rays to next wall
        wall_intersection: np.ndarray, shape = (n_stpes, n_wall_rays)
            point of intersection for each wall ray with wall
        """
        start_locomotion = np.array(start_locomotion)

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

        # get initial model center_head distance
        dis_CH = getDistance(*start_position)

        # format oritracks = [center_x, center_y, orientation]
        oritracks = np.empty((n_steps, 3))
        binned_locs_out = np.empty((n_steps, 3), dtype=np.int32)
        wall_distances = np.empty((n_steps, self.n_wall_rays))
        wall_intersections = np.empty((n_steps, self.n_wall_rays, 2))

        # get first oritrack
        prev_oritracks = np.empty(3)
        prev_oritracks[0:2] = start_position[[2, 3]]
        prev_oritracks[2] = getAngle(
            (1, 0),
            (
                start_position[0] - start_position[2],
                start_position[1] - start_position[3],
            ),
            "radians",
        )

        for idx_step in range(n_steps):
            # 1. get wall rays
            (
                binned_wall_rays,
                wall_distance,
                wall_intersection,
            ) = self._get_binned_wall_ray(prev_oritracks)

            # 2. get fish model prediction
            with torch.no_grad():
                logit_lin, logit_ang, logit_ori = self.model(
                    *prev_loc[:, None].to(self.device),
                    binned_wall_rays[None, :]
                )
            # 3. convert model output into locomotion
            probs_lin = F.softmax(logit_lin[0] / temperature, -1)
            probs_ang = F.softmax(logit_ang[0] / temperature, -1)
            probs_ori = F.softmax(logit_ori[0] / temperature, -1)

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

            # 4. update output lists
            oritracks[idx_step] = prev_oritracks
            wall_distances[idx_step] = wall_distance
            wall_intersections[idx_step] = wall_intersection
            binned_locs_out[idx_step] = (
                binned_lin,
                binned_ang,
                binned_ori,
            )
            prev_loc = torch.tensor((binned_lin, binned_ang, binned_ori))

            # 5. update tracks
            unbinned_loc = unbin_loc(
                binned_locomotion=prev_loc[None, :],
                n_bins_lin=self.n_bins_lin,
                n_bins_ang=self.n_bins_ang,
                n_bins_ori=self.n_bins_ori,
            )[0]

            # 6. update oritracks
            prev_oritracks = row_l2c(prev_oritracks, unbinned_loc)

        # convert oritracks to normal tracks
        tracks = convPolarToCart(oritracks, [dis_CH])

        return binned_locs_out, tracks, wall_distances, wall_intersections


def run_simulation(
    config: Dict[str, Union[str, float, int]],
    model: FishWallRays,
    device: torch.device,
    start_position: Sequence,
    start_locomotion: Sequence,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_position = np.array(start_position)
    start_locomotion = np.array(start_locomotion)

    # get sim
    sim = SimulationWallRays(
        model=model,
        n_bins_lin=model.n_bins_lin,
        n_bins_ang=model.n_bins_ang,
        n_bins_ori=model.n_bins_ori,
        n_bins_wall_rays=model.n_bins_wall_rays,
        n_wall_rays=model.n_wall_rays,
        field_of_view=model.field_of_view,
        egocentric_wall_ray_orientations=model.egocentric_wall_ray_orientations,
        max_view=model.max_view,
        device=device,
    )

    # run
    _, tracks, wall_distances, wall_intersections = sim.run(
        config["n_steps"],
        start_locomotion=start_locomotion,
        start_position=start_position,
        sample=config["sample"],
        temperature=config["temperature"],
    )

    return tracks, wall_distances, wall_intersections
