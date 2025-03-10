import os
from typing import Dict, List, Union, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.load import load_locomotions
from src.locomotion import unbin_loc, convLocToCart, bin_loc
from src.visualization import addTracksOnTank
from src.torch_datasets import SimpleDataset
from src.models.fish_simple import FishSimple
from src.simulations.simulation_simple import SimulationSimple
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
    model: FishSimple,
    device: torch.device,
    n_steps: int,
    path_vid_out: str,
    start_positions: Sequence,
    start_locomotion: Sequence,
) -> None:
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
        n_steps,
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

    # add on video
    addTracksOnTank(
        path_output_video=path_vid_out,
        tracks=tracks,
        nfish=1,
        skeleton=[(0, 1)],
        fish_point_size=[1, 2],
    )


def get_dataset(
    config: Dict[str, Union[float, str, int]]
) -> Tuple[Dataset, Dataset]:
    loc = load_locomotions(config["path_tracks"])
    # bin locomotion
    binned_loc = bin_loc(
        loc, config["n_bins_lin"], config["n_bins_ang"], config["n_bins_ori"]
    )
    # separate fish
    data_locs = list()
    label_locs = list()
    for f in range(3):
        fish_loc = binned_loc[:, 3 * f : 3 * f + 3]
        data_locs.append(fish_loc[:-1])
        label_locs.append(fish_loc[1:])
    # concat
    inputs = np.concatenate(data_locs, axis=0)
    labels = np.concatenate(label_locs, axis=0)
    data = np.concatenate((inputs, labels), axis=1)

    # split into train and test
    n_test = int(data.shape[0] * (1 - config["data_split"]))
    # sample randomly
    rng = np.random.default_rng(config["seed"])
    idcs_test = rng.choice(list(range(data.shape[0])), n_test, replace=False)

    test_data = data[idcs_test]
    train_data = np.delete(data, idcs_test, axis=0)

    return SimpleDataset(train_data), SimpleDataset(test_data)


def collate_fn(
    batch: List[np.ndarray],
) -> Tuple[
    Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    data = np.stack(batch, axis=0)
    train_data = {
        "id_lin": data[:, 0],
        "id_ang": data[:, 1],
        "id_ori": data[:, 2],
    }
    test_data = (data[:, 3], data[:, 4], data[:, 5])
    return (to_tensor(train_data), to_tensor(test_data, targets=True))


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
    model = FishSimple(config)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    log.info("Loading data")
    train_ds, test_ds = get_dataset(config)

    train_dl = get_dataloader(config, train_ds, shuffle=True)
    test_dl = get_dataloader(config, test_ds, shuffle=False)

    loss_eval = eval(test_dl, model, device)
    log.info(f"Initial eval loss: {loss_eval:2.4f}")
    start_positions = np.array([647.72, 121.83, 625.18, 115.30])
    eval_vid_path = os.path.join(config["visual_eval_dir"], "before.mp4")
    visualize_simulation(
        config,
        model,
        device,
        n_steps=300,
        path_vid_out=eval_vid_path,
        start_positions=start_positions,
        start_locomotion=[1, 0, 0],
    )
    log.info(f"Visual evaluation saved to {eval_vid_path}")
    log.info("Starting training")
    for e in range(1, config["n_epochs"] + 1):
        loss_train = epoch(train_dl, model, optim, device)
        loss_eval = eval(test_dl, model, device)
        log.info(f"{e:3} | Train: {loss_train:2.4f} | Test: {loss_eval:2.4f}")
    log.info(f"Final eval loss: {loss_eval:2.4f}")
    eval_vid_path = os.path.join(config["visual_eval_dir"], "after.mp4")
    visualize_simulation(
        config,
        model,
        device,
        n_steps=300,
        path_vid_out=eval_vid_path,
        start_positions=start_positions,
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
        # model
        "embed_dim_lin": 100,
        "embed_dim_ang": 100,
        "embed_dim_ori": 100,
        "hidden": 210,
        "dropout": 0.1,
        "model_dir": "models/simple_fish_v1",
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
