import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import cv2

import load


def addTracksOnVideo(
    inputvideo,
    outputvideo,
    tracks,
    nfish=3,
    fps=30,
    dimension=(960, 720),
    psize=1,
    showvid=False,
    skeleton=None,
):
    """
    Takes tracks and adds them on video
    skeleton is a mapping indices for each point in tracks
    Fishskeleton: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,4), (3,5), (2,6), (3,7), (6,8), (7,8), (8,9)]
    Only Center and Head: [(0,1)]
    """
    row, col = tracks.shape
    assert row >= 1
    assert col > 0
    assert col % nfish == 0
    nnodes = col // nfish

    # Set up input
    cap = cv2.VideoCapture(inputvideo)
    if cap is None:
        print("not able to open video")
        sys.exit(-1)

    # Set up output
    fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
    out = cv2.VideoWriter(outputvideo, fourcc, fps, dimension)

    # Process video
    colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]
    i = 0
    while cap.isOpened():
        if i % 1000 == 0:
            print("Frame: ", i)
        if i == row:
            print("trackset too short")
            sys.exit(-1)

        success, frame = cap.read()
        if success:
            # frame = cv2.flip( frame, 0 )
            for f in range(nfish):
                points = []
                for n in range(nnodes // 2):
                    x = int(tracks[i, f * nnodes + 2 * n])
                    y = int(tracks[i, f * nnodes + 2 * n + 1])
                    points.append(
                        (
                            x,
                            y,
                        )
                    )
                for p in points:
                    frame = cv2.circle(frame, p, psize, colors[f], -1)
                # Lines between points
                if skeleton is not None:
                    for p1, p2 in skeleton:
                        frame = cv2.line(
                            frame, points[p1], points[p2], colors[f], 1
                        )
            out.write(frame)
            if showvid:
                cv2.imshow("fishy fish fish", frame)
                cv2.waitKey(0)
        else:
            print("end of video")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        i += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if i < row - 1:
        print("Not the entire trackset was used")


def addRaycasts(
    raycasts: np.ndarray,
    frame: np.ndarray,
    center: Tuple[int, int],
    max_view_range: int,
    color: Tuple[int, int, int],
) -> np.ndarray:
    for ray in raycasts[:21]:
        if ray != 0:
            frame = cv2.ellipse(
                frame,
                center,
                (max_view_range, max_view_range),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=color,
            )
    return cv2.ellipse(
        frame,
        center,
        (max_view_range, max_view_range),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=color,
    )


def addTracksOnTank(
    path_output_video: str,
    tracks: np.ndarray,
    path_tank_img: str = "data/tank.png",
    nfish: int = 3,
    fps: int = 30,
    dimension: Tuple[int, int] = (960, 720),
    fish_point_size: int = 1,
    show_video_during_rendering: bool = False,
    skeleton: List[Tuple[int, int]] = None,
    path_raycasts: str = None,
):
    """
    Takes tracks and adds them on video
    skeleton is a mapping to create lines between points.
    Lines are created between indices for each point.
    Fishskeleton: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,4), (3,5), (2,6), (3,7), (6,8), (7,8), (8,9)]
    Only Center and Head: [(0,1)]
    """
    row, col = tracks.shape
    assert row >= 1
    assert col > 0
    assert col % nfish == 0
    nnodes = col // nfish

    # Set up input
    img = cv2.imread(path_tank_img, 1)
    if img is None:
        print("not able to open tank")
        sys.exit(-1)

    # load raycasts if given
    if path_raycasts is not None:
        raycasts = load.load_raycasts(path_raycasts)
        count_bin_agents = 21
        wall_ray_walls = 15
        radius_field_of_view_walls = 180
        radius_field_of_view_agents = 300
        max_view_range = 90
        count_fishes = 3

    # Set up output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path_output_video, fourcc, fps, dimension)

    # Process video
    colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]
    i = 0
    while i < row:
        if i % 1000 == 0:
            print("Frame: ", i)

        frame = img.copy()
        for f in range(nfish):
            points = []
            for n in range(nnodes // 2):
                x = int(tracks[i, f * nnodes + 2 * n])
                y = int(tracks[i, f * nnodes + 2 * n + 1])
                points.append((x, y))
            for p in points:
                frame = cv2.circle(frame, p, fish_point_size, colors[f], -1)
            # Lines between points
            if skeleton is not None:
                for p1, p2 in skeleton:
                    frame = cv2.line(
                        frame, points[p1], points[p2], colors[f], 1
                    )
            # raycasts
            if path_raycasts is not None:
                indices = list(range(f * 21, (f + 1) * 21)) + list(
                    range(21 * 3 + f * 15, 21 * 3 + (f + 1) * 15)
                )
                frame = addRaycasts(
                    raycasts.to_numpy()[i, indices],
                    frame,
                    points[1],
                    max_view_range,
                    colors[f],
                )
        out.write(frame)
        if show_video_during_rendering:
            cv2.imshow("fishy fish fish", frame)
            cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        i += 1

    out.release()
    cv2.destroyAllWindows()


def main():
    # addTracksOnVideo( diff2, diff2_out, tracks, showvid=True, skeleton=[(0,1), (0,2), (0,3), (1,2), (1,3), (2,4), (3,5), (2,6), (3,7), (6,8), (7,8), (8,9)] )
    import reader

    tracks = reader.extract_coordinates(
        "data/sleap/diff1.h5",
        [b"head", b"center"],
    )[0:1000]
    addTracksOnTank(
        "videos/dev.mp4",
        tracks,
        show_video_during_rendering=True,
        skeleton=[(0, 1)],
        path_raycasts="data/raycast/diff1.csv",
    )
    pass


if __name__ == "__main__":
    main()
