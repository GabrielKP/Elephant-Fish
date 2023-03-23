import numpy as np

import reader
import locomotion
import visualization


def test_diff1_loc():
    tracks = reader.extract_coordinates(
        "data/sleap/diff1.h5",
        [
            b"head",
            b"center",
            # b"l_fin_basis",
            # b"r_fin_basis",
            # b"l_fin_end",
            # b"r_fin_end",
            # b"l_body",
            # b"r_body",
            # b"tail_basis",
            # b"tail_end",
        ],
    )[0:3000]

    # convert to locs
    locs = locomotion.getnLoc(tracks, nnodes=1, nfish=3)

    startpoints = np.array(
        [
            647.728515625,
            121.83802032470703,
            625.1812133789062,
            115.30104064941406,
            354.48193359375,
            70.24964141845703,
            341.4715270996094,
            90.89854431152344,
            389.50885009765625,
            126.94833374023438,
            402.483642578125,
            115.75283813476562,
        ]
    )

    # convert back to cartesian
    tracks_locs = locomotion.convLocToCart(locs, startpoints)

    visualization.addTracksOnTank(
        "videos/test/diff1_loc.mp4",
        tracks_locs,
        showvid=False,
        skeleton=[
            (0, 1),
            # (0, 2),
            # (0, 3),
            # (1, 2),
            # (1, 3),
            # (2, 4),
            # (3, 5),
            # (2, 6),
            # (3, 7),
            # (6, 8),
            # (7, 8),
            # (8, 9),
        ],
    )
    visualization.addTracksOnTank(
        "videos/test/diff1_sleap.mp4",
        tracks,
        showvid=False,
        skeleton=[(0, 1)],
    )


if __name__ == "__main__":
    test_diff1_loc()
