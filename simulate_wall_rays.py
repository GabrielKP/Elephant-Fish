from typing import Dict

import numpy as np

from src.utils import check_make_dirs, get_logger, get_device
from src.models.fish_wall_rays import FishWallRays
from src.simulations.simulation_wall_rays import run_simulation
from src.visualization import addTracksOnTank

log = get_logger(__name__)


def simulate(config: Dict):
    device = get_device(config)

    log.info(f"Loading model from {config['model_dir']}.")
    model = FishWallRays.from_dir(config["model_dir"])
    model.eval()

    log.info("Running Simulation.")
    tracks, wall_distances, wall_intersections = run_simulation(
        config=config,
        model=model,
        device=device,
        start_locomotion=config["start_locomotion"],
        start_position=config["start_position"],
    )

    log.info("Rendering video.")
    addTracksOnTank(
        path_output_video=config["path_output_video"],
        tracks=tracks,
        nfish=1,
        skeleton=[(0, 1)],
        fish_point_size=[1, 2],
        wall_distances=wall_distances,
        wall_intersections=wall_intersections,
        config=config,
    )


if __name__ == "__main__":
    config = {
        # model
        "model_dir": "models/wr_fish_v1",
        "device": "cpu",
        # output dir
        "path_output_video": "videos/simulation/wr_v1.mp4",
        # simulation args
        "max_view": 200,
        "n_steps": 1000,
        "sample": True,
        "temperature": 0.6,
        "start_position": [647.72, 121.83, 625.18, 115.30],
        "start_locomotion": [1, 0, 0],
    }
    simulate(config)
