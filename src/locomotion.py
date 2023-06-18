from typing import Tuple, List

import numpy as np

from src.functions import (
    getAngle,
    getDistance,
    convPolarToCart,
    get_distances,
    getAngles,
    getDistances,
)


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
            [head1_x, head1_y, center1_x, center1_y, ...]
            [head1_x, head1_y, center1_x, center1_y, ...]
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
                   given as distance within the used coordinate system
            * ang: angle from current to next center position (egocentric)
                   given as radians: [0,2pi)
            * ori: change in orientation from current to next center position
                   given as radians: [0,2pi)

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


def get_bins(bins_range: Tuple[int, int], n_bins: int) -> np.ndarray:
    return np.linspace(
        start=bins_range[0],
        stop=bins_range[1],
        num=n_bins - 1,
        endpoint=True,
    )


def bin_loc(
    locomotion: np.ndarray,
    n_bins_lin: int,
    n_bins_ang: int,
    n_bins_ori: int,
    bins_range_lin: Tuple[int, int] = (-7, 13),
) -> np.ndarray:
    """
    Returns binned locs

    Parameters
    ----------
    locomotion: np.ndarray, shape = (n_rows, n_fish * 3)
        locomotion data, n_rows is the amount of "locomotion steps".
        [
            [r0_f1_lin, r0_f1_ang, r0_f1_ori, r0_f2_lin, ...],
            [r1_f1_lin, r1_f1_ang, r1_f1_ori, r1_f2_lin, ...],
            ...
        ]
    n_bins_lin: int
    n_bins_ang: int
    n_bins_ori: int
        amount of bins for each movement type
    bins_range_lin: (int, int), default = (-7, 13)
        range for which bins are created, one bin is allotted for
        values lower or higher than the threshold respectively.
        based on plots determined most linear movement lies
        between -7 and + 13.

    Returns
    -------
    binned_loc: np.ndarray, shape = (n_rows, n_fish * 3)
        binned locomotion, encoded as indices of one hot vectors
        [
            [r0_f1_lin_bin_id, r0_f1_ang_bin_id, r0_f1_ori_bin_id, r0_f2_lin_bin_id, ...],
            [r1_f1_lin_bin_id, r1_f1_ang_bin_id, r1_f1_ori_bin_id, r1_f2_lin_bin_id, ...],
            ...
        ]
    """
    nfish = locomotion.shape[-1] // 3

    # 1. Get indices and output array
    binned_loc = np.empty((locomotion.shape), np.int32)
    # locs indices
    lin = [3 * x for x in range(nfish)]
    ang = [3 * x + 1 for x in range(nfish)]
    ori = [3 * x + 2 for x in range(nfish)]

    # 2. Linear
    bins_lin = get_bins(bins_range_lin, n_bins_lin)
    binned_loc[:, lin] = np.digitize(locomotion[:, lin], bins_lin)
    # 3. Angular
    # (do not require one extra bin as all values are [0,2pi),
    #  thus can create more within the linspace)
    bins_ang = get_bins((0, 2 * np.pi), n_bins_ang + 1)
    binned_loc[:, ang] = np.digitize(locomotion[:, ang], bins_ang)
    # 4. Orientation
    bins_ori = get_bins((0, 2 * np.pi), n_bins_ori + 1)
    binned_loc[:, ori] = np.digitize(locomotion[:, ori], bins_ori)

    return binned_loc


def unbin_loc(
    binned_locomotion: np.ndarray,
    n_bins_lin: int,
    n_bins_ang: int,
    n_bins_ori: int,
    bins_range_lin: Tuple[int, int] = (-7, 13),
) -> np.ndarray:
    """
    Returns binned locs as normal locs by replacing bin values with
    center bin value.

    Parameters
    ----------
    binned_loc: np.ndarray, shape = (n_rows, n_fish * 3)
        binned locomotion, encoded as indices.
        n_rows is the amount of "locomotion steps".
        [
            [r0_f1_lin_bin_id, r0_f1_ang_bin_id, r0_f1_ori_bin_id, r0_f2_lin_bin_id, ...],
            [r1_f1_lin_bin_id, r1_f1_ang_bin_id, r1_f1_ori_bin_id, r1_f2_lin_bin_id, ...],
            ...
        ]
    n_bins_lin: int
    n_bins_ang: int
    n_bins_ori: int
        amount of bins for each movement type
    bins_range_lin: (int, int), default = (-7, 13)
        range for which bins were created.

    Returns
    -------
    locomotion: np.ndarray, shape = (n_rows, n_fish * 3)
        locomotion data
        [
            [r0_f1_lin, r0_f1_ang, r0_f1_ori, r0_f2_lin, ...],
            [r1_f1_lin, r1_f1_ang, r1_f1_ori, r1_f2_lin, ...],
            ...
        ]
    """
    nfish = binned_locomotion.shape[-1] // 3

    # 1. Get indices and output array
    locomotion = np.empty((binned_locomotion.shape))
    # locs indices
    lin = [3 * x for x in range(nfish)]
    ang = [3 * x + 1 for x in range(nfish)]
    ori = [3 * x + 2 for x in range(nfish)]

    # 2. linear
    # get mean bin values
    bin_vals_lin = np.empty(n_bins_lin)
    bins_lin = get_bins(bins_range_lin, n_bins_lin)
    # add mean of bin distance to each bin start
    bin_dis_lin = (bins_lin[1:] - bins_lin[:-1]) / 2
    bin_vals_lin[1:-1] = bins_lin[:-1] + bin_dis_lin
    # for edge cases add/subtract previous bin mean double (arbitrary)
    bin_vals_lin[0] = bins_lin[0] - 2 * bin_dis_lin[0]
    bin_vals_lin[-1] = bins_lin[-1] + 2 * bin_dis_lin[-1]
    # convert
    locomotion[:, lin] = bin_vals_lin[binned_locomotion[:, lin]]

    # 3. angular
    bin_vals_ang = np.empty(n_bins_ang)
    bins_ang = get_bins((0, 2 * np.pi), n_bins_ang + 1)
    # add mean of bin distance to each bin start
    bin_dis_ang = (bins_ang[1:] - bins_ang[:-1]) / 2
    bin_vals_ang[1:] = bins_ang[:-1] + bin_dis_ang
    bin_vals_ang[0] = 0  # values in bin 0 cannot be below or above 0 => 0
    # convert
    locomotion[:, ang] = bin_vals_ang[binned_locomotion[:, ang]]

    # 3. orientation
    bin_vals_ori = np.empty(n_bins_ori)
    bins_ori = get_bins((0, 2 * np.pi), n_bins_ori + 1)
    # add mean of bin distance to each bin start
    bin_dis_ori = (bins_ori[1:] - bins_ori[:-1]) / 2
    bin_vals_ori[1:] = bins_ori[:-1] + bin_dis_ori
    bin_vals_ori[0] = 0  # values in bin 0 cannot be below or above 0 => 0
    # convert
    locomotion[:, ori] = bin_vals_ori[binned_locomotion[:, ori]]

    return locomotion


if __name__ == "__main__":
    pass
