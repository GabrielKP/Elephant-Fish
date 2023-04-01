import os
from typing import Dict, Union, Sequence

import numpy as np
import torch

import locomotion
import visualization
import evaluation
from utils import check_make_dirs, get_device
from model import FishSimple
from simulation import SimulationSimple


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
    unbinned_loc = locomotion.unbin_loc(
        binned_locomotion=binned_loc,
        n_bins_lin=model.n_bins_lin,
        n_bins_ang=model.n_bins_ang,
        n_bins_ori=model.n_bins_ori,
    )

    # to cartesian
    tracks = locomotion.convLocToCart(
        unbinned_loc,
        start_positions,
    )

    return tracks


def evaluate(config: Dict[str, Union[str, float, bool]]):
    device = get_device(config)
    model = FishSimple.from_dir(config["model_dir"])
    # run simulation
    start_positions = np.array([647.72, 121.83, 625.18, 115.30])
    start_locomotion = np.array([1, 0, 0])
    tracks = run_simulation(
        config, model, device, start_positions, start_locomotion
    )
    # get evaluation graphs
    evaluation.create_plots(
        tracks, direc=os.path.join(config["model_dir"], "plots")
    )
    # put tracks on video
    if config.get("visualize", False):
        if config.get(config["visual_eval_dir"], None) is not None:
            path_output_video = config["visual_eval_dir"]
        else:
            path_output_video = os.path.join(config["model_dir"], "video")
        path_output_video = path_output_video.replace(
            "<model_dir>", config["model_dir"]
        )
        check_make_dirs(path_output_video)
        visualization.addTracksOnTank(
            path_output_video=path_output_video,
            tracks=tracks,
            nfish=1,
            skeleton=[(0, 1)],
            fish_point_size=[1, 2],
        )


if __name__ == "__main__":
    config = {
        "model_dir": "models/simple_fish_v1",
        # simulation
        "device": "cpu",
        "n_steps": 1000,
        "sample": True,
        "temperature": 0.6,
        # visualization
        "visualize": True,
        "visual_eval_dir": None,  # default: model_dir/video
    }
    evaluate(config)
