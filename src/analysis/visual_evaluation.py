import os
from typing import Dict, Union, Sequence

import numpy as np

from src.visualization import addTracksOnTank
from src.analysis.plot import create_plots
from src.utils import check_make_dirs, get_device
from src.models.fish_simple import FishSimple
from src.simulations.simulation_simple import run_simulation


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
    create_plots(tracks, direc=os.path.join(config["model_dir"], "plots"))
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
        addTracksOnTank(
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
