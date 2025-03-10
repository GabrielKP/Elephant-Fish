import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn
import numpy as np
import pandas as pd

from src.analysis.utils import calc_follow, calc_iid, calc_tlvc
from src.functions import convertRadiansRange, get_indices, readClusters
from src.reader import extract_coordinates
from src.locomotion import getnLoc
from src.utils import check_make_dirs


def plot_follow(tracks, max_tolerated_movement=20, multipletracksets=False):
    """
    Create and save Follow graph, only use center nodes for it
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    follow = []
    if not multipletracksets:
        assert tracks.shape[-1] % 2 == 0
        nfish = int(tracks.shape[-1] / 2)

        # for every fish combination calculate the follow
        for i1 in range(nfish):
            for i2 in range(i1 + 1, nfish):
                f1_x, f1_y = get_indices(i1)
                f2_x, f2_y = get_indices(i2)
                follow.append(
                    calc_follow(
                        tracks[:, f1_x : f1_y + 1], tracks[:, f2_x : f2_y + 1]
                    )
                )
                follow.append(
                    calc_follow(
                        tracks[:, f2_x : f2_y + 1], tracks[:, f1_x : f1_y + 1]
                    )
                )
    else:
        for trackset in tracks:
            assert trackset.shape[-1] % 2 == 0
            nfish = int(trackset.shape[-1] / 2)

            for i1 in range(nfish):
                for i2 in range(i1 + 1, nfish):
                    f1_x, f1_y = get_indices(i1)
                    f2_x, f2_y = get_indices(i2)
                    follow.append(
                        calc_follow(
                            trackset[:, f1_x : f1_y + 1],
                            trackset[:, f2_x : f2_y + 1],
                        )
                    )
                    follow.append(
                        calc_follow(
                            trackset[:, f2_x : f2_y + 1],
                            trackset[:, f1_x : f1_y + 1],
                        )
                    )

    follow = np.concatenate(follow, axis=0)

    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.93)
    ax.set_xlim(-max_tolerated_movement, max_tolerated_movement)
    seaborn.distplot(
        pd.Series(follow, name="Follow"),
        ax=ax,
        hist_kws={"rwidth": 0.9, "color": "y"},
    )

    return fig


def plot_iid(tracks, multipletracksets=False):
    """
    Create and save iid graph, only use center nodes for it
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    iid = []
    if not multipletracksets:
        assert tracks.shape[-1] % 2 == 0
        nfish = int(tracks.shape[-1] / 2)
        # for every fish combination calculate iid
        for i1 in range(nfish):
            for i2 in range(i1 + 1, nfish):
                f1_x, f1_y = get_indices(i1)
                f2_x, f2_y = get_indices(i2)
                iid.append(
                    calc_iid(
                        tracks[:-1, f1_x : f1_y + 1],
                        tracks[:-1, f2_x : f2_y + 1],
                    )
                )
                iid.append(iid[-1])
    else:
        for trackset in tracks:
            assert trackset.shape[-1] % 2 == 0
            nfish = int(trackset.shape[-1] / 2)
            # for every fish combination calculate iid
            for i1 in range(nfish):
                for i2 in range(i1 + 1, nfish):
                    f1_x, f1_y = get_indices(i1)
                    f2_x, f2_y = get_indices(i2)
                    iid.append(
                        calc_iid(
                            trackset[:-1, f1_x : f1_y + 1],
                            trackset[:-1, f2_x : f2_y + 1],
                        )
                    )
                    iid.append(iid[-1])

    iid = np.concatenate(iid, axis=0)

    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.93)
    ax.set_xlim(0, 700)
    seaborn.distplot(
        pd.Series(iid, name="IID [pixel]"),
        ax=ax,
        hist_kws={"rwidth": 0.9, "color": "y"},
    )

    return fig


def plot_follow_iid(tracks, multipletracksets=False):
    """
    plots fancy graph with follow and iid, only use with center values
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    copied from Moritz Maxeiner
    """
    follow = []
    iid = []

    if not multipletracksets:
        assert tracks.shape[-1] % 2 == 0
        nfish = int(tracks.shape[-1] / 2)
        # for every fish combination calculate the follow
        for i1 in range(nfish):
            for i2 in range(i1 + 1, nfish):
                f1_x, f1_y = get_indices(i1)
                f2_x, f2_y = get_indices(i2)
                iid.append(
                    calc_iid(
                        tracks[:-1, f1_x : f1_y + 1],
                        tracks[:-1, f2_x : f2_y + 1],
                    )
                )
                iid.append(iid[-1])

                follow.append(
                    calc_follow(
                        tracks[:, f1_x : f1_y + 1], tracks[:, f2_x : f2_y + 1]
                    )
                )
                follow.append(
                    calc_follow(
                        tracks[:, f2_x : f2_y + 1], tracks[:, f1_x : f1_y + 1]
                    )
                )
    else:
        for trackset in tracks:
            assert trackset.shape[-1] % 2 == 0
            nfish = int(trackset.shape[-1] / 2)
            for i1 in range(nfish):
                for i2 in range(i1 + 1, nfish):
                    f1_x, f1_y = get_indices(i1)
                    f2_x, f2_y = get_indices(i2)
                    iid.append(
                        calc_iid(
                            trackset[:-1, f1_x : f1_y + 1],
                            trackset[:-1, f2_x : f2_y + 1],
                        )
                    )
                    iid.append(iid[-1])

                    follow.append(
                        calc_follow(
                            trackset[:, f1_x : f1_y + 1],
                            trackset[:, f2_x : f2_y + 1],
                        )
                    )
                    follow.append(
                        calc_follow(
                            trackset[:, f2_x : f2_y + 1],
                            trackset[:, f1_x : f1_y + 1],
                        )
                    )

    follow_iid_data = pd.DataFrame(
        {
            "IID [pixel]": np.concatenate(iid, axis=0),
            "Follow": np.concatenate(follow, axis=0),
        }
    )

    grid = seaborn.jointplot(
        x="IID [pixel]",
        y="Follow",
        data=follow_iid_data,
        linewidth=0,
        s=1,
        kind="scatter",
    )
    grid.ax_joint.set_xlim(0, 700)
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.subplots_adjust(top=0.9)
    return grid.fig


def plot_tlvc_iid(
    tracks,
    time_step=(1000 / 30),
    tau_seconds=(0.3, 1.3),
    multipletracksets=False,
):
    """
    TLVC_IDD by Moritz Maxeiner
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    tau_min_seconds, tau_max_seconds = tau_seconds

    tau_min_frames = int(tau_min_seconds * 1000.0 / time_step)
    tau_max_frames = int(tau_max_seconds * 1000.0 / time_step)

    tlvc = []
    iid = []

    if not multipletracksets:
        assert tracks.shape[-1] % 2 == 0
        nfish = int(tracks.shape[-1] / 2)
        # for every fish combination calculate the follow
        for i1 in range(nfish):
            for i2 in range(i1 + 1, nfish):
                f1_x, f1_y = get_indices(i1)
                f2_x, f2_y = get_indices(i2)
                iid.append(
                    calc_iid(
                        tracks[1 : -tau_max_frames + 1, f1_x : f1_y + 1],
                        tracks[1 : -tau_max_frames + 1, f2_x : f2_y + 1],
                    )
                )
                iid.append(iid[-1])

                a_v = tracks[1:, f1_x : f1_y + 1] - tracks[:-1, f1_x : f1_y + 1]
                b_v = tracks[1:, f2_x : f2_y + 1] - tracks[:-1, f2_x : f2_y + 1]
                tlvc.append(calc_tlvc(a_v, b_v, tau_min_frames, tau_max_frames))
                tlvc.append(calc_tlvc(b_v, a_v, tau_min_frames, tau_max_frames))
    else:
        for trackset in tracks:
            assert trackset.shape[-1] % 2 == 0
            nfish = int(trackset.shape[-1] / 2)
            # for every fish combination calculate the follow
            for i1 in range(nfish):
                for i2 in range(i1 + 1, nfish):
                    f1_x, f1_y = get_indices(i1)
                    f2_x, f2_y = get_indices(i2)
                    iid.append(
                        calc_iid(
                            trackset[1 : -tau_max_frames + 1, f1_x : f1_y + 1],
                            trackset[1 : -tau_max_frames + 1, f2_x : f2_y + 1],
                        )
                    )
                    iid.append(iid[-1])

                    a_v = (
                        trackset[1:, f1_x : f1_y + 1]
                        - trackset[:-1, f1_x : f1_y + 1]
                    )
                    b_v = (
                        trackset[1:, f2_x : f2_y + 1]
                        - trackset[:-1, f2_x : f2_y + 1]
                    )
                    tlvc.append(
                        calc_tlvc(a_v, b_v, tau_min_frames, tau_max_frames)
                    )
                    tlvc.append(
                        calc_tlvc(b_v, a_v, tau_min_frames, tau_max_frames)
                    )

    tlvc_iid_data = pd.DataFrame(
        {
            "IID [pixel]": np.concatenate(iid, axis=0),
            "TLVC": np.concatenate(tlvc, axis=0),
        }
    )

    grid = seaborn.jointplot(
        x="IID [pixel]",
        y="TLVC",
        data=tlvc_iid_data,
        linewidth=0,
        s=1,
        kind="scatter",
    )
    grid.ax_joint.set_xlim(0, 700)
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.subplots_adjust(top=0.9)
    return grid.fig


def plot_tankpositions(tracks, multipletracksets=False):
    """
    Heatmap of fishpositions
    By Moritz Maxeiner
    """
    x_pos = []
    y_pos = []

    if not multipletracksets:
        assert tracks.shape[-1] % 2 == 0
        nfish = int(tracks.shape[-1] / 2)
        for i1 in range(nfish):
            f1_x, f1_y = get_indices(i1)
            x_pos.append(tracks[:, f1_x])
            y_pos.append(tracks[:, f1_y])
    else:
        for trackset in tracks:
            assert trackset.shape[-1] % 2 == 0
            nfish = int(trackset.shape[-1] / 2)
            for i1 in range(nfish):
                f1_x, f1_y = get_indices(i1)
                x_pos.append(trackset[:, f1_x])
                y_pos.append(trackset[:, f1_y])

    x_pos = np.concatenate(x_pos, axis=0)
    y_pos = np.concatenate(y_pos, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(top=0.91)
    ax.set_xlim(0, 960)
    ax.set_ylim(0, 720)
    seaborn.kdeplot(x=x_pos, y=y_pos, n_levels=25, shade=True, ax=ax)
    return fig


def plot_velocities(tracks, clusterfile=None, multipletracksets=False):
    """
    Plots the velocities
    Expects two nodes per fish exactly: tracks: [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
                                                [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
                                                ...
    """

    if clusterfile is not None:
        cLin, cAng, cOri = readClusters(clusterfile)

    if not multipletracksets:
        assert tracks.shape[-1] % 4 == 0
        nfish = int(tracks.shape[-1] / 4)

        locs = getnLoc(tracks, nnodes=1, nfish=nfish)

        # Get dem indices
        i_lin = [x * 3 for x in range(nfish)]
        i_ang = [x * 3 + 1 for x in range(nfish)]
        i_trn = [x * 3 + 2 for x in range(nfish)]
        linear_velocities = locs[:, i_lin]
        angular_velocities = locs[:, i_ang]
        turn_velocities = locs[:, i_trn]
    else:
        linear_velocities = []
        angular_velocities = []
        turn_velocities = []
        for trackset in tracks:
            assert trackset.shape[-1] % 4 == 0
            nfish = int(trackset.shape[-1] / 4)

            locs = getnLoc(tracks, nnodes=1, nfish=nfish)

            # Get dem indices
            i_lin = [x * 3 for x in range(nfish)]
            i_ang = [x * 3 + 1 for x in range(nfish)]
            i_trn = [x * 3 + 2 for x in range(nfish)]
            linear_velocities.append(np.concatenate(locs[:, i_lin]))
            angular_velocities.append(np.concatenate(locs[:, i_ang]))
            turn_velocities.append(np.concatenate(locs[:, i_trn]))

    angular_velocities = np.concatenate(angular_velocities, axis=0)
    linear_velocities = np.concatenate(linear_velocities, axis=0)
    turn_velocities = np.concatenate(turn_velocities, axis=0)

    angular_velocities = convertRadiansRange(angular_velocities)
    turn_velocities = convertRadiansRange(turn_velocities)

    fig_angular, ax = plt.subplots(figsize=(18, 18))
    fig_angular.subplots_adjust(top=0.93)
    ax.set_xlim(-np.pi, np.pi)
    seaborn.distplot(
        pd.Series(angular_velocities, name="Angular movement"),
        ax=ax,
        hist_kws={"rwidth": 0.9, "color": "y"},
    )
    if clusterfile is not None:
        seaborn.rugplot(cAng, height=0.03, ax=ax, color="r", linewidth=3)

    fig_turn, ax = plt.subplots(figsize=(18, 18))
    fig_turn.subplots_adjust(top=0.93)
    ax.set_xlim(-np.pi, np.pi)
    seaborn.distplot(
        pd.Series(turn_velocities, name="Orientational movement"),
        ax=ax,
        hist_kws={"rwidth": 0.9, "color": "y"},
    )
    if clusterfile is not None:
        seaborn.rugplot(cOri, height=0.03, ax=ax, color="r", linewidth=3)

    fig_linear, ax = plt.subplots(figsize=(18, 18))
    fig_linear.subplots_adjust(top=0.93)
    ax.set_xlim(-20, 20)
    seaborn.distplot(
        pd.Series(linear_velocities, name="Linear movement"),
        ax=ax,
        hist_kws={"rwidth": 0.9, "color": "y"},
    )
    if clusterfile is not None:
        seaborn.rugplot(cLin, height=0.03, ax=ax, color="r", linewidth=3)

    return fig_linear, fig_angular, fig_turn


def plot_trajectories(tracks, world=(960, 720)):
    """
    Plots tank trajectory of fishes
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    data = {
        fish: pd.DataFrame(
            {
                "x": tracks[:, fish * 2],
                "y": tracks[:, fish * 2 + 1],
            }
        )
        for fish in range(nfish)
    }
    combined_data = pd.concat(
        [data[fish].assign(Agent=f"Agent {fish}") for fish in data.keys()]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    seaborn.set_style("white", {"axes.linewidth": 2, "axes.edgecolor": "black"})

    seaborn.scatterplot(
        x="x", y="y", hue="Agent", linewidth=0, s=16, data=combined_data, ax=ax
    )
    ax.set_xlim(0, world[0])
    ax.set_ylim(0, world[1])
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")

    ax.scatter(
        [frame["x"][0] for frame in data.values()],
        [frame["y"][0] for frame in data.values()],
        marker="h",
        c="black",
        s=64,
        label="Start",
    )
    ax.scatter(
        [frame["x"][len(frame["x"]) - 1] for frame in data.values()],
        [frame["y"][len(frame["y"]) - 1] for frame in data.values()],
        marker="x",
        c="black",
        s=64,
        label="End",
    )
    ax.legend()

    return fig


def create_plots(
    tracks,
    direc="figures/latest_plots",
    time_step=(1000 / 30),
    tau_seconds=(0.3, 1.3),
    clusterfile="data/clusters.txt",
):
    """Create evaluation plots for given tracks

    Parameters
    ----------
    tracks : np.ndarray, shape = (n_rows, 4 * nfish)
        Cartesian coordinates of head node and center node per fish:
        [
            [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...],
            [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...],
            ...
        ]
    dir : str
        dir in which to save the plots
    time_step : float, default = 1000/30
    tau_seconds : (float, float), default = (0.3, 1.3)
    clusterfile : str, default = "data/clusters.txt"
    """
    assert tracks.shape[-1] % 4 == 0
    nfish = int(tracks.shape[-1] / 4)

    # handle dir
    check_make_dirs(direc)

    # Extract Center nodes
    i_center_values = [x for x in range(nfish * 4) if x % 4 > 1]
    tracksCenter = tracks[:, i_center_values]

    # make and save graphs
    if nfish > 1:
        save_figure(plot_iid(tracksCenter), path=os.path.join(direc, "iid.png"))
        save_figure(
            plot_follow(tracksCenter), path=os.path.join(direc, "follow.png")
        )
        save_figure(
            plot_follow_iid(tracksCenter),
            path=os.path.join(direc, "follow_iid.png"),
        )
        save_figure(
            plot_tlvc_iid(tracksCenter, time_step, tau_seconds),
            path=os.path.join(direc, "tlvc_iid.png"),
        )
    save_figure(
        plot_tankpositions(tracksCenter),
        path=os.path.join(direc, "tankpostions.png"),
        size=(24, 18),
    )
    # Velocities
    lin, ang, trn = plot_velocities(tracks, clusterfile=clusterfile)
    save_figure(
        lin, path=os.path.join(direc, "locomotion_linear.png"), size=(18, 18)
    )
    save_figure(
        ang, path=os.path.join(direc, "locomotion_angular.png"), size=(18, 18)
    )
    save_figure(
        trn, path=os.path.join(direc, "locomotion_trn.png"), size=(18, 18)
    )
    # Trajectories
    save_figure(
        plot_trajectories(tracksCenter),
        path=os.path.join(direc, "trajectories_all.png"),
        size=(24, 18),
    )
    # Print trajectories for each fish
    if nfish != 1:
        for f in range(nfish):
            fx, fy = get_indices(f)
            save_figure(
                plot_trajectories(tracksCenter[:, [fx, fy]]),
                path=os.path.join(
                    direc, "trajectories_agent" + str(f) + ".png"
                ),
                size=(24, 18),
            )


def create_all_plots_together(
    path="figures/together", clusterfile="data/clusters.txt"
):
    """
    creates plots for all fishdata together
    """

    # handle dir
    if not os.path.isdir(path):
        # create dir
        try:
            os.mkdir(path)
        except OSError:
            print("Dir Creation failed")
    if path[-1] != "/":
        path = path + "/"

    # Load data
    tracks1 = extract_coordinates("data/sleap_1_diff2.h5", [b"head", b"center"])
    tracks2 = extract_coordinates("data/sleap_1_diff2.h5", [b"head", b"center"])
    tracks3 = extract_coordinates(
        "data/sleap_1_diff3.h5", [b"head", b"center"]
    )[0:17000]
    tracks4 = extract_coordinates(
        "data/sleap_1_diff4.h5", [b"head", b"center"]
    )[120:]
    tracks5 = extract_coordinates("data/sleap_1_same1.h5", [b"head", b"center"])
    tracks6 = extract_coordinates(
        "data/sleap_1_same3.h5", [b"head", b"center"]
    )[130:]
    tracks7 = extract_coordinates("data/sleap_1_same4.h5", [b"head", b"center"])
    tracks8 = extract_coordinates("data/sleap_1_same5.h5", [b"head", b"center"])

    # Get Centerpoints
    nfish = 3
    isc = [x for x in range(nfish * 4) if x % 4 > 1]
    tracks1c = tracks1[:, isc]
    tracks2c = tracks2[:, isc]
    tracks3c = tracks3[:, isc]
    tracks4c = tracks4[:, isc]
    tracks5c = tracks5[:, isc]
    tracks6c = tracks6[:, isc]
    tracks7c = tracks7[:, isc]
    tracks8c = tracks8[:, isc]

    print("follow_iid")
    save_figure(
        plot_follow_iid(
            [
                tracks1c,
                tracks2c,
                tracks3c,
                tracks4c,
                tracks5c,
                tracks6c,
                tracks7c,
                tracks8c,
            ],
            multipletracksets=True,
        ),
        path=(path + "follow_iid.png"),
    )
    print("tlvc_iid")
    save_figure(
        plot_tlvc_iid(
            [
                tracks1c,
                tracks2c,
                tracks3c,
                tracks4c,
                tracks5c,
                tracks6c,
                tracks7c,
                tracks8c,
            ],
            multipletracksets=True,
        ),
        path=(path + "tlvc_iid.png"),
    )
    print("follow")
    save_figure(
        plot_follow(
            [
                tracks1c,
                tracks2c,
                tracks3c,
                tracks4c,
                tracks5c,
                tracks6c,
                tracks7c,
                tracks8c,
            ],
            multipletracksets=True,
        ),
        path=(path + "follow.png"),
    )
    print("iid")
    save_figure(
        plot_iid(
            [
                tracks1c,
                tracks2c,
                tracks3c,
                tracks4c,
                tracks5c,
                tracks6c,
                tracks7c,
                tracks8c,
            ],
            multipletracksets=True,
        ),
        path=(path + "iid.png"),
    )
    print("tankpositions")
    save_figure(
        plot_tankpositions(
            [
                tracks1c,
                tracks2c,
                tracks3c,
                tracks4c,
                tracks5c,
                tracks6c,
                tracks7c,
                tracks8c,
            ],
            multipletracksets=True,
        ),
        path=(path + "tankpositions.png"),
        size=(24, 18),
    )
    print("movements/velocities/locomotions")
    lin, ang, ori = plot_velocities(
        [
            tracks1,
            tracks2,
            tracks3,
            tracks4,
            tracks5,
            tracks6,
            tracks7,
            tracks8,
        ],
        clusterfile=clusterfile,
        multipletracksets=True,
    )
    # lin, ang, ori = plot_velocities( [tracks, tracks2], multipletracksets=True )
    save_figure(lin, path=(path + "locomotion_linear.png"), size=(18, 18))
    save_figure(ang, path=(path + "locomotion_angular.png"), size=(18, 18))
    save_figure(ori, path=(path + "locomotion_orientation.png"), size=(18, 18))


def create_all_plots_separate(clusterfile="data/clusters.txt"):
    """
    Creates plots for every video seperatly
    """
    tracks1 = extract_coordinates("data/sleap_1_diff1.h5", [b"head", b"center"])
    tracks2 = extract_coordinates("data/sleap_1_diff2.h5", [b"head", b"center"])
    tracks3 = extract_coordinates(
        "data/sleap_1_diff3.h5", [b"head", b"center"]
    )[0:17000]
    tracks4 = extract_coordinates(
        "data/sleap_1_diff4.h5", [b"head", b"center"]
    )[120:]
    tracks5 = extract_coordinates("data/sleap_1_same1.h5", [b"head", b"center"])
    tracks6 = extract_coordinates(
        "data/sleap_1_same3.h5", [b"head", b"center"]
    )[130:]
    tracks7 = extract_coordinates("data/sleap_1_same4.h5", [b"head", b"center"])
    tracks8 = extract_coordinates("data/sleap_1_same5.h5", [b"head", b"center"])

    create_plots(tracks1, path="figures/diff1", clusterfile="data/clusters.txt")
    create_plots(tracks2, path="figures/diff2", clusterfile="data/clusters.txt")
    create_plots(tracks3, path="figures/diff3", clusterfile="data/clusters.txt")
    create_plots(tracks4, path="figures/diff4", clusterfile="data/clusters.txt")
    create_plots(tracks5, path="figures/same1", clusterfile="data/clusters.txt")
    create_plots(tracks6, path="figures/same3", clusterfile="data/clusters.txt")
    create_plots(tracks7, path="figures/same4", clusterfile="data/clusters.txt")
    create_plots(tracks8, path="figures/same5", clusterfile="data/clusters.txt")


def save_figure(fig, path="figures/latest_plot.png", size=(25, 12.5)):
    """
    Saves the given figure in path with certain size
    """
    x, y = size
    fig.set_size_inches(x, y)
    fig.savefig(path)
    plt.close(fig)


def animate_positions(track, track2=None):
    """
    Animation of all postions. Not optimized.
    """

    frames, positions = track.shape

    assert positions % 2 == 0

    i_x = list(range(0, positions, 2))
    i_y = list(range(1, positions, 2))

    fig = plt.figure()

    def update_points(n, track, points):
        points.set_xdata(track[n, i_x])
        points.set_ydata(track[n, i_y])

    def update_points2(n, track, track2, points, points2):
        points.set_xdata(track[n, i_x])
        points.set_ydata(track[n, i_y])
        points2.set_xdata(track2[n, i_x])
        points2.set_ydata(track2[n, i_y])

    plt.xlim(0, 960)
    plt.ylim(9, 720)

    (points,) = plt.plot([], [], "r.")
    if track2 is None:
        point_animation = animation.FuncAnimation(
            fig,
            update_points,
            track.shape[0],
            fargs=(track, points),
            interval=10,
        )
    else:
        (points2,) = plt.plot([], [], "b.")
        point_animation = animation.FuncAnimation(
            fig,
            update_points2,
            track.shape[0],
            fargs=(track, track2, points, points2),
            interval=10,
        )

    plt.show()


def plot_train_history(history, title):
    """
    Plot history object from keras.
    From: https://www.tensorflow.org/tutorials/structured_data/time_series
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.legend()

    plt.show()


def main():
    create_all_plots_together()
    create_all_plots_separate()


if __name__ == "__main__":
    main()
