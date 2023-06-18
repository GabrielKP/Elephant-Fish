from typing import Tuple, Optional

import numpy as np

from src.reader import extract_coordinates
from src.visualization import addTracksOnTank
from src.functions import getAngles
from src.utils import get_bins

# lines are given as (point1,point2)
TANK_BORDERS = np.array(
    [
        [[183, 0], [183, 720]],  # left
        [[183, 147], [270, 63]],  # left -> top
        [[0, 63], [960, 63]],  # top
        [[672, 63], [762, 153]],  # top -> right
        [[762, 0], [762, 720]],  # right
        [[762, 522], [678, 639]],  # right -> bot
        [[0, 639], [960, 639]],  # bot
        [[285, 639], [183, 528]],  # bot -> left
    ]
)
# shape = (4, 2, 2)


def perp(vec: np.ndarray) -> np.ndarray:
    # vec.shape = (2)
    vec_perp = np.empty_like(vec)
    vec_perp[0] = -vec[1]
    vec_perp[1] = vec[0]
    # vec_perp.shape = (2)
    return vec_perp


def vec_dot(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    # vec1.shape = (n_positions, n_wall_rays, 2)
    # vec2.shape = (n_positions, 2) OR (2)
    # shape = (n_positions, n_wall_rays)
    return np.sum(vec1 * vec2, axis=-1)


def vectorized_intersect(
    points1_a: np.ndarray,
    points2_a: np.ndarray,
    points1_b: np.ndarray,
    points2_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # https://web.archive.org/web/20111108065352/https://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
    # points1_a.shape = (2) - border_start
    # points2_a.shape = (2) - border_end
    # points1_b.shape = (n_positions, 1, 2) - fish
    # points2_b.shape = (n_positions, n_wall_rays, 2) - wall_ray

    # vec border_start -> border_end
    vec_a = points2_a - points1_a  # shape = (2)
    # vec fish -> wall ray
    vec_b = points2_b - points1_b  # shape = (n_positions, n_wall_rays, 2)
    # vec fish -> border_start
    vec_a1b1 = points1_a - points1_b  # shape = (n_positions, 1, 2)
    vec_a_perp = perp(vec_a)  # shape = (2)
    denominator = vec_dot(
        vec_a_perp, vec_b
    )  # shape = (n_positions, n_wall_rays)
    numerator = vec_dot(vec_a_perp, vec_a1b1)  # shape = (n_positions, 1)
    multiplier = numerator / denominator  # shape = (n_positions, n_wall_rays)
    smultiplier = multiplier[
        :, :, None
    ]  # shape = (n_positions, n_wall_rays, 1)

    intersect_points = (
        smultiplier * vec_b + points1_b
    )  # shape = (n_positions, n_wall_rays, 2)

    # also return positions where multiplier is negative or nan
    negative_nan_smultiplier = smultiplier < 0 | np.isnan(
        smultiplier
    )  # shape = (n_positions, n_wall_rays, 1)
    return intersect_points, negative_nan_smultiplier


def get_raycasts(
    current_positions: np.ndarray,
    orientations: np.ndarray,
    egocentric_wall_ray_orientations: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # current_positions.shape = (n_positions, 2) -> center_x, center_y
    # orientations = (n_positions)
    # egocentric_wall_rays_orientations = (n_wallrays)
    n_wall_rays = egocentric_wall_ray_orientations.shape[0]
    n_positions = current_positions.shape[0]

    # get allocentric wall ray orientations
    allocentric_wall_ray_orientations = (
        orientations[:, None] + egocentric_wall_ray_orientations[None, :]
    ) % (2 * np.pi)

    ## get arbitrary point in direction of wall ray (to have a line)
    wall_rays_point2_x = np.cos(allocentric_wall_ray_orientations)
    wall_rays_point2_y = np.sin(allocentric_wall_ray_orientations)
    # shape = (n_positions, n_wallrays)
    egocentric_wall_rays_point2 = (
        np.stack((wall_rays_point2_x, wall_rays_point2_y), axis=-1) * 10
    )
    # shape = (n_positions, n_wallrays, 2)
    # allocentric points on wall rays
    allocentric_wall_rays_point2 = (
        current_positions[:, None] + egocentric_wall_rays_point2
    )  # shape = (n_positions, n_wallrays, 2)

    # compute intersects for every wall line and every ray independently
    border_intersects = list()
    for idx_wall in range(TANK_BORDERS.shape[0]):
        points1_wall = TANK_BORDERS[idx_wall, 0]  # (2)
        points2_wall = TANK_BORDERS[idx_wall, 1]  # (2)

        # current positions acts as point1
        points1_wall_ray = current_positions[
            :, None
        ]  # shape = (n_positions, 1, 2)
        points2_wall_ray = (
            allocentric_wall_rays_point2  # shape = (n_positions, n_wallrays, 2)
        )
        intersects, negative_nan_smultiplier = vectorized_intersect(
            points1_a=points1_wall,
            points2_a=points2_wall,
            points1_b=points1_wall_ray,
            points2_b=points2_wall_ray,
        )  # shape = ((n_positions, n_wallrays, 2),(n_positions, n_wallrays, 2))

        # disregard negative and nan intersects
        negative_nan_smultiplier = np.concatenate(
            (negative_nan_smultiplier, negative_nan_smultiplier), axis=-1
        )
        intersects[negative_nan_smultiplier] = np.nan

        border_intersects.append(intersects)

    border_intersects = np.stack(border_intersects, axis=2)
    # shape = (n_positions, n_wallrays, n_borders, 2)

    # get vector from center to intersect for every border
    # (should be along the wall rays)
    vecs_wall_ray_all_borders = (
        current_positions[:, None, None] - border_intersects
    )
    # shape = (n_positions, n_wallrays, n_borders, 2)
    # TODO: if dot(vecs_wall_ray_all_borders * allocentric_wallray) < 1, discard
    # because it is the wall "behind" the wall ray

    # get distances
    powered = vecs_wall_ray_all_borders**2
    vec_lengths = np.sqrt(powered[:, :, :, 0] + powered[:, :, :, 1])
    # shape = (n_positions, n_wallrays, n_borders)

    # for each position, for each wallray, get nearest border
    nearest_wall = np.nanargmin(vec_lengths, axis=2)
    # shape = (n_positions, n_wallrays)

    # select correct distance
    wall_distances = vec_lengths[
        np.arange(n_positions)[:, None],
        np.arange(n_wall_rays)[None, :],
        nearest_wall,
    ]
    # shape = (n_positions, n_wallrays)

    # select intersections
    wall_intersections = border_intersects[
        np.arange(n_positions)[:, None],
        np.arange(n_wall_rays)[None, :],
        nearest_wall,
    ]
    # shape = (n_positions, n_wallrays, 2)

    return nearest_wall, wall_distances, wall_intersections


def raycasts_from_tracks(
    tracks: np.ndarray,
    egocentric_wall_ray_orientations: np.ndarray,
    orientations: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    tracks : np.ndarray, shape = (n_positions, 4)
        Trackset, expects head and center :
        [
            [head1_x, head1_y, center1_x, center1_x]
            ...
        ]
    egocentric_wall_ray_orientations : np.ndarray, shape = (n_wall_rays)
        Vector or radians describing directions of wall rays
        [wall_ray1_ori, wall_ray2_ori, ...]
    orientations : np.ndarray, optional, default = `None`, shape(n_positions)
        When orientations are passed, they are not computed additionally,
        and tracks[:,0:2] are unused.

    Returns
    -------
    nearest_wall : np.ndarray, shape = (n_positions, n_wallrays)
        index for which wall is the one which cross the wall ray first
    wall_distances : np.ndarray, shape = (n_positions, n_wallrays)

    """
    n_positions = tracks.shape[0]

    # get fish orientations at each point
    if orientations is None:
        vec_look = tracks[:, 0:2] - tracks[:, 2:4]
        allocentric_vec = np.empty((n_positions, 2))
        allocentric_vec.fill(0)
        allocentric_vec[:, 0] = 1
        orientations = getAngles(
            allocentric_vec, vec_look
        )  # shape = (n_positions)

    center_positions = tracks[:, 2:4]

    nearest_wall, wall_distances, wall_intersections = get_raycasts(
        center_positions, orientations, egocentric_wall_ray_orientations
    )

    # nearest_wall.shape = (n_positions, n_wallrays)
    # wall_distances.shape = (n_positions, n_wallrays)
    # wall_intersections.shape = (n_positions, n_wallrays, 2)
    return nearest_wall, wall_distances, wall_intersections


def get_egocentric_wall_ray_orientations(
    n_wallrays: int,
    field_of_view: Tuple[float, float],
) -> np.ndarray:
    """Returns egocentric wall ray orientations in radians.

    Parameters
    ----------
    n_wallrays : int
        Amount of wall rays
    field_of_view : (float, float)
        Wall rays are distributed along this field of view.
        First value is mininum, second is maximum.
        Values are converted to radians [0, 2*pi).
    """

    egocentric_wall_ray_orientations = np.linspace(
        start=field_of_view[0], stop=field_of_view[1], num=n_wallrays
    )

    egocentric_wall_ray_orientations = egocentric_wall_ray_orientations % (
        2 * np.pi
    )

    # shape = (n_wallrays)
    return egocentric_wall_ray_orientations


def bin_wall_rays(
    wall_distances: np.ndarray,
    n_bins_wall_rays: int,
    bins_range_wall_rays: Tuple[float, float],
) -> np.ndarray:
    # wall_distances.shape = (n_positions, n_wall_rays)

    bins = get_bins(bins_range_wall_rays, n_bins_wall_rays)
    binned_wall_rays = np.digitize(wall_distances, bins)

    # binned_wall_rays.shape = (n_positions, n_wall_rays)
    return binned_wall_rays


def unbin_wall_rays(
    binned_wall_rays: np.ndarray,
    n_bins_wall_rays: int,
    bins_range_wall_rays: Tuple[float, float],
) -> np.ndarray:
    # binned_wall_rays.shape = (n_positions, n_wall_rays)

    bin_vals = np.empty(n_bins_wall_rays)
    bins = get_bins(bins_range_wall_rays, n_bins_wall_rays)
    # add mean of bin distance to each bin start
    bin_dis = (bins[1:] - bins[:-1]) / 2
    bin_vals[1:-1] = bins[:-1] + bin_dis
    # for edge cases add/subtract previous bin mean double (arbitrary)
    bin_vals[0] = bins[0] - 2 * bin_dis[0]
    bin_vals[-1] = bins[-1] + 2 * bin_dis[-1]
    # convert bins into values
    wall_distances = bin_vals[binned_wall_rays]

    # wall_distances.shape = (n_positions, n_wall_rays)
    return wall_distances


def main():
    n_wallrays = 15
    field_of_view = (3 / 4 * -np.pi, 3 / 4 * np.pi)

    tracks = extract_coordinates(
        "data/sleap/diff1.h5",
        [
            b"head",
            b"center",
        ],
        fish_to_extract=[0],
    )[:1000]

    egocentric_wall_ray_orientations = get_egocentric_wall_ray_orientations(
        n_wallrays, field_of_view
    )

    nearest_wall, wall_distances, wall_intersections = raycasts_from_tracks(
        tracks, egocentric_wall_ray_orientations
    )

    n_bins_wall_rays = 90
    bins_range_wall_rays = (0.0, 200)
    binned_wall_distances = bin_wall_rays(
        wall_distances,
        n_bins_wall_rays,
        bins_range_wall_rays,
    )

    wall_distances = unbin_wall_rays(
        binned_wall_distances,
        n_bins_wall_rays,
        bins_range_wall_rays,
    )

    addTracksOnTank(
        "videos/test/raycast.mp4",
        tracks,
        nfish=1,
        skeleton=[(0, 1)],
        wall_intersections=wall_intersections,
        wall_distances=wall_distances,
        config={"max_view": 200},
    )


if __name__ == "__main__":
    main()
