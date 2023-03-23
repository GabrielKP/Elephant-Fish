import math
from itertools import chain
from typing import Optional

import numpy as np
import pandas as pd

from functions import (
    getAngle,
    getDistance,
    readClusters,
    distancesToClusters,
    softmax,
    get_indices,
    convPolarToCart,
    get_distances,
    getAngles,
    getDistances,
    convertAngle,
)
from reader import *


def convertLocmotionToBin(
    loco, clusters_path, path_to_save=None, probabilities=True
):
    # get cluster centers
    clusters_mov, clusters_pos, clusters_ori = readClusters(clusters_path)

    result = None
    # convert locomotion into bin representation for each fish
    for i in range(0, int(loco.shape[1] / 3)):
        if probabilities:
            # compute distances to cluster centers and invert them (1/x) (exp so we do not get divide by zero)
            dist_mov = 1 / np.exp(
                distancesToClusters(loco[:, i * 3], clusters_mov)
            )
            dist_pos = 1 / np.exp(
                distancesToClusters(loco[:, i * 3 + 1], clusters_pos)
            )
            dist_ori = 1 / np.exp(
                distancesToClusters(loco[:, i * 3 + 2], clusters_ori)
            )

            # get probabilites row-wise with softmax function and append header
            prob_mov = np.append(
                np.array(
                    [
                        [
                            "Fish_" + str(i) + "_prob_next_x_bin_" + str(j)
                            for j in range(0, len(clusters_mov))
                        ]
                    ]
                ),
                softmax(dist_mov),
                axis=0,
            )
            prob_pos = np.append(
                np.array(
                    [
                        [
                            "Fish_" + str(i) + "_prob_next_y_bin_" + str(j)
                            for j in range(0, len(clusters_pos))
                        ]
                    ]
                ),
                softmax(dist_pos),
                axis=0,
            )
            prob_ori = np.append(
                np.array(
                    [
                        [
                            "Fish_" + str(i) + "_prob_ori_bin_" + str(j)
                            for j in range(0, len(clusters_ori))
                        ]
                    ]
                ),
                softmax(dist_ori),
                axis=0,
            )

            temp = np.append(
                np.append(prob_mov, prob_pos, axis=1), prob_ori, axis=1
            )
            if i == 0:
                result = temp
            else:
                result = np.append(result, temp, axis=1)
        else:
            # todo
            pass

    if path_to_save == None:
        return result[1:]
    else:
        df = pd.DataFrame(data=result[1:], columns=result[0])
        df.to_csv(path_to_save, sep=";")


def row_l2c(coords: np.ndarray, locs: np.ndarray) -> np.ndarray:
    """
    Returns 1d ndarray with new coordinates based on previos coordinades and given locomotions,
    Output: [center1_x, center1_y, orientation1, ...]
    """
    nfish = len(coords) // 3

    # coords indices
    xs = [3 * x for x in range(nfish)]
    ys = [3 * x + 1 for x in range(nfish)]
    os = [3 * x + 2 for x in range(nfish)]
    # locs indices
    lin = [3 * x for x in range(nfish)]
    ang = [3 * x + 1 for x in range(nfish)]
    ori = [3 * x + 2 for x in range(nfish)]
    # computation
    new_angles = (coords[os] + locs[ang]) % (np.pi * 2)
    xvals = np.cos(new_angles) * np.abs(locs[lin])
    yvals = np.sin(new_angles) * np.abs(locs[lin])
    out = np.empty(coords.shape)
    out[xs] = coords[xs] + xvals
    out[ys] = coords[ys] + yvals
    out[os] = (coords[os] + locs[ori]) % (np.pi * 2)
    return out


def convLocToCart(loc: np.ndarray, startpoints: List[float]) -> np.ndarray:
    """
    Converts locomotion np array to coordinates
    loc:
        2d array, per row 3 entries per fish, [linear movement, angular movement, orientation movement]:
        [
            [fish1_lin, fish1_ang, fish1_trn, fish2_lin, fish2_ang, fish2_trn, ...]
            [fish1_lin, fish1_ang, fish1_trn, fish2_lin, fish2_ang, fish2_trn, ...]
            ...
        ]
    startpoints:
        two nodes per fish exactly:
        [head1_x, head1_y, center1_x, center1_y, head2_x, head2_y, center2_x, center2_y,...]
    Output:

        [
            [center1_x, center1_y, orientation1, ...]
            [center1_x, center1_y, orientation1, ...]
            ...
        ]
    """
    row, col = loc.shape
    assert row > 1
    assert col % 3 == 0
    nfish = col // 3
    assert len(startpoints) // nfish == 4

    # save [center1_x, center2_y, orientation1, center2_x, ...] for every fish
    out = np.empty([row + 1, nfish * 3])

    # 1. Distances Center - Head, out setup
    disCH = []
    for f in range(nfish):
        disCH.append(
            getDistance(
                startpoints[4 * f],
                startpoints[4 * f + 1],
                startpoints[4 * f + 2],
                startpoints[4 * f + 3],
            )
        )
        out[0, 3 * f] = startpoints[4 * f + 2]
        out[0, 3 * f + 1] = startpoints[4 * f + 3]
        # Angle between Fish Orientation and the unit vector
        out[0, 3 * f + 2] = getAngle(
            (
                1,
                0,
            ),
            (
                startpoints[4 * f] - startpoints[4 * f + 2],
                startpoints[4 * f + 1] - startpoints[4 * f + 3],
            ),
            "radians",
        )

    for i in range(0, row):
        out[i + 1] = row_l2c(out[i], loc[i])

    return convPolarToCart(out, disCH)


def getnLoc(tracks: np.ndarray, nnodes: int, nfish: int = 3) -> np.ndarray:
    """
    Computes Locomotion for n nodes

    Parameters
    ----------
    tracks: np.ndarray
        Trackset, expects head and center at least:
        [
            [head1_x, head2_y, center1_x, center1_x, ..., head2_x, ...]
            ...
        ]
    nnodes: int >= 1
        Amount of nodes per fish in output.
        For every node > 1 an extra distance and angle with respect to
        the center node is created.

    Returns
    -------
    locomotion : np.ndarray, shape: [n_fish * 3, n_rows_tracks - 1]
        Vector containing for every fish:
            * lin: distance from current to next center position
            * ang: angle from current to next center position (egocentric)
            * ori: change in orientation from current to next center position

        nnodes=1:
        [
            [r0_f1_lin, r0_f1_ang, r0_f1_ori, r0_f2_lin,...],
            [r1_f1_lin, r1_f1_ang, r1_f1_ori, r1_f2_lin,...],
            ...
        ]
        nnodes=3:
        [
            [r0_f1_lin, r0_f1_ang, r0_f1_ori, r0_f1_head_dis, r0_f1_head_ang, r0_f1_node3_dis, r0_f1_node3_ang, r0_f2_lin,...],
            [r1_f1_lin, r1_f1_ang, r1_f1_ori, r1_f1_head_dis, r1_f1_head_ang, r1_f1_node3_dis, r1_f1_node3_ang, r1_f2_lin,...],
            ...
        ]
    """
    rows, cols = tracks.shape
    assert cols >= 4 * nfish
    assert rows > 1
    assert nnodes >= 1

    nf = nnodes * 2 + 1  # loc entries per fish
    out = np.empty((rows - 1, nf * nfish))
    for f in range(nfish):
        ## Set first 3 entries
        head_next = tracks[1:, [4 * f, 4 * f + 1]]
        center_next = tracks[1:, [4 * f + 2, 4 * f + 3]]
        # head - center
        vec_look = (
            tracks[:-1, [4 * f, 4 * f + 1]]
            - tracks[:-1, [4 * f + 2, 4 * f + 3]]
        )
        # head - center
        vec_look_next = head_next - center_next
        # center_next - center
        vec_next = center_next - tracks[:-1, [4 * f + 2, 4 * f + 3]]
        out[:, nf * f] = get_distances(tracks[:, [4 * f + 2, 4 * f + 3]])[:, 0]
        out[:, nf * f + 1] = getAngles(vec_look, vec_next)
        out[:, nf * f + 2] = getAngles(vec_look, vec_look_next)
        ## Set every other node in relation to orientation and center node
        for n in range(nnodes - 1):
            # since head is at the first position...
            ix = nf * f + 3
            if n == 0:
                out[:, ix + 2 * n] = getDistances(center_next, head_next)
                # Since the new orientation is exactly the angle o the vector between head and center
                out[:, ix + 2 * n + 1] = 0
            else:
                node_next = tracks[
                    1:, [4 * f + 2 + 2 * n, 4 * f + 2 + 2 * n + 1]
                ]
                vec_cn_next = node_next - center_next
                out[:, ix + 2 * n] = getDistances(center_next, node_next)
                out[:, ix + 2 * n + 1] = getAngles(vec_look_next, vec_cn_next)

    return out


if __name__ == "__main__":
    pass
