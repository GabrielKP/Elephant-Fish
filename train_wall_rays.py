import os
from typing import Dict, List, Union, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.load import load_locomotions
from src.raycasts import (
    bin_wall_rays,
    get_egocentric_wall_ray_orientations,
    raycasts_from_tracks,
)
from src.reader import extract_coordinates
from src.locomotion import bin_loc, getnLoc
from src.visualization import addTracksOnTank
from src.torch_datasets import WallRayDataset
from src.models.fish_wall_rays import FishWallRays
from src.simulations.simulation_wall_rays import run_simulation
from src.utils import get_logger, to_device, get_device, to_tensor


log = get_logger(__name__)


def get_loss(
    model: nn.Module,
    device: torch.device,
    batch: Tuple[Dict, Tuple],
) -> torch.Tensor:
    train_data, labels = batch
    logits_lin, logits_ang, logits_ori = model(**to_device(train_data, device))

    labels_lin, labels_ang, labels_ori = to_device(labels, device)

    loss_lin = F.cross_entropy(logits_lin, labels_lin)
    loss_ang = F.cross_entropy(logits_ang, labels_ang)
    loss_ori = F.cross_entropy(logits_ori, labels_ori)
    return loss_lin + loss_ang + loss_ori


def eval(
    test_dl: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        eval_loss: float = 0
        for batch in test_dl:
            loss_total = get_loss(model, device, batch)
            eval_loss += loss_total.detach().cpu().item()
        return eval_loss / len(test_dl)


def epoch(
    train_dl: DataLoader,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    epoch_loss: float = 0
    for batch in train_dl:
        optim.zero_grad()

        loss_total = get_loss(model, device, batch)
        loss_total.backward()

        optim.step()
        epoch_loss += loss_total.detach().cpu().item()
    return epoch_loss / len(train_dl)


def visualize_simulation(
    config: Dict,
    model: FishWallRays,
    device: torch.device,
    n_steps: int,
    path_vid_out: str,
    start_position: Sequence,
    start_locomotion: Sequence,
) -> None:
    start_position = np.array(start_position)
    start_locomotion = np.array(start_locomotion)

    tracks, wall_distances, wall_intersections = run_simulation(
        {**config, "n_steps": n_steps},
        model=model,
        device=device,
        start_locomotion=start_locomotion,
        start_position=start_position,
    )

    # add on video
    addTracksOnTank(
        path_output_video=path_vid_out,
        tracks=tracks,
        nfish=1,
        skeleton=[(0, 1)],
        fish_point_size=[1, 2],
        wall_distances=wall_distances,
        wall_intersections=wall_intersections,
        config=config,
    )


def get_dataset(
    config: Dict[str, Union[float, str, int]]
) -> Tuple[Dataset, Dataset]:
    # load tracks
    tracks_multi_fish = extract_coordinates(
        config["path_tracks"],
        [b"head", b"center"],
    )
    # merge multiple fish tracks into one
    tracks = list()
    for f in range(3):
        tracks.append(tracks_multi_fish[:, 4 * f : 4 * f + 4])
    tracks = np.concatenate(tracks, axis=0)

    # get binned locomotion
    loc = getnLoc(tracks, 1, nfish=1)
    binned_loc = bin_loc(
        loc, config["n_bins_lin"], config["n_bins_ang"], config["n_bins_ori"]
    )
    # get binned wall distances
    egocentric_wall_ray_orientations = get_egocentric_wall_ray_orientations(
        config["n_wall_rays"], config["field_of_view"]
    )
    _, wall_distances, _ = raycasts_from_tracks(
        tracks, egocentric_wall_ray_orientations
    )
    binned_wall_distances = bin_wall_rays(
        wall_distances,
        config["n_bins_wall_rays"],
        (0.0, config["max_view"]),
    )

    # organize
    loc_data = binned_loc[:-1]
    wr_data = binned_wall_distances[:-1]
    loc_label = binned_loc[1:]

    # split into train and test
    n_test = int(loc_data.shape[0] * (1 - config["data_split"]))
    # sample randomly
    rng = np.random.default_rng(config["seed"])
    idcs_test = rng.choice(
        list(range(loc_data.shape[0])), n_test, replace=False
    )

    test_data_loc = loc_data[idcs_test]
    test_data_wr = wr_data[idcs_test]
    test_label_loc = loc_label[idcs_test]
    train_data_loc = np.delete(loc_data, idcs_test, axis=0)
    train_data_wr = np.delete(wr_data, idcs_test, axis=0)
    train_label_loc = np.delete(loc_label, idcs_test, axis=0)

    return WallRayDataset(
        train_data_loc, train_data_wr, train_label_loc
    ), WallRayDataset(test_data_loc, test_data_wr, test_label_loc)


def collate_fn(
    batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[
    Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    locs, wrs, labels = zip(*batch)
    locs = np.stack(locs, axis=0)
    wrs = np.stack(wrs, axis=0)
    labels = np.stack(labels, axis=0)
    inputs_ = {
        "id_lin": locs[:, 0],
        "id_ang": locs[:, 1],
        "id_ori": locs[:, 2],
        "ids_wall_ray": wrs,
    }
    labels = (labels[:, 0], labels[:, 1], labels[:, 2])
    return (to_tensor(inputs_), to_tensor(labels, targets=True))


def get_dataloader(
    config: Dict[str, Union[float, str, int]], dataset: Dataset, **kwargs
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        **kwargs,
    )


def train(config: Dict[str, Union[float, str, int]]):
    torch.manual_seed(config["seed"])

    device = get_device(config)

    log.info("Loading model")
    model = FishWallRays(config)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    log.info("Loading data")
    train_ds, test_ds = get_dataset(config)

    train_dl = get_dataloader(config, train_ds, shuffle=True)
    test_dl = get_dataloader(config, test_ds, shuffle=False)

    loss_eval = eval(test_dl, model, device)
    log.info(f"Initial eval loss: {loss_eval:2.4f}")
    start_position = np.array([647.72, 121.83, 625.18, 115.30])
    eval_vid_path = os.path.join(config["visual_eval_dir"], "wr_before.mp4")
    visualize_simulation(
        config,
        model,
        device,
        n_steps=300,
        path_vid_out=eval_vid_path,
        start_position=start_position,
        start_locomotion=[1, 0, 0],
    )
    log.info(f"Visual evaluation saved to {eval_vid_path}")
    log.info("Starting training")
    for e in range(1, config["n_epochs"] + 1):
        loss_train = epoch(train_dl, model, optim, device)
        loss_eval = eval(test_dl, model, device)
        log.info(f"{e:3} | Train: {loss_train:2.4f} | Test: {loss_eval:2.4f}")
    log.info(f"Final eval loss: {loss_eval:2.4f}")
    eval_vid_path = os.path.join(config["visual_eval_dir"], "wr_after.mp4")
    visualize_simulation(
        config,
        model,
        device,
        n_steps=300,
        path_vid_out=eval_vid_path,
        start_position=start_position,
        start_locomotion=[1, 0, 0],
    )
    log.info(f"Visual evaluation saved to {eval_vid_path}")

    model.save(config)

    return model


if __name__ == "__main__":
    config = {
        "path_tracks": "data/sleap/diff1.h5",
        "data_split": 0.8,
        # locomotion
        "n_bins_lin": 300,
        "n_bins_ang": 300,
        "n_bins_ori": 300,
        # wall rays
        "n_wall_rays": 15,
        "n_bins_wall_rays": 90,
        "field_of_view": (3 / 4 * -np.pi, 3 / 4 * np.pi),
        "max_view": 200,
        # model
        "embed_dim_lin": 100,
        "embed_dim_ang": 100,
        "embed_dim_ori": 100,
        "embed_dim_wall_rays": 30,
        "hidden": 210,
        "dropout": 0.1,
        "model_dir": "models/wr_fish_v1",
        # training
        "n_epochs": 3,
        "batch_size": 24,
        "lr": 0.001,
        "seed": 123,
        "device": "cpu",
        # visual eval
        "visual_eval_dir": "videos/eval/",
        "sample": True,
        "temperature": 0.6,
    }
    train(config)
