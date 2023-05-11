import os
import sys
from typing import List, Dict, Tuple, Union

import numpy as np
import cv2
from tqdm import trange


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


def addTracksOnTank(
    path_output_video: str,
    tracks: np.ndarray,
    path_tank_img: str = "data/tank.png",
    nfish: int = 3,
    fps: int = 30,
    dimension: Tuple[int, int] = (960, 720),
    fish_point_size: Union[int, List[int]] = 1,
    show_video_during_rendering: bool = False,
    skeleton: List[Tuple[int, int]] = None,
    wall_intersections: np.ndarray = None,
    wall_distances: np.ndarray = None,
    config: Dict = {},
):
    """
    Takes tracks and adds them on video
    skeleton is a mapping to create lines between points.
    Lines are created between indices for each point.
    Fishskeleton: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,4), (3,5), (2,6), (3,7), (6,8), (7,8), (8,9)]
    Only Center and Head: [(0,1)]
    fish_point_size: int | List[int]
        either same size for every point, or the sizes
        for each point in order of nodes
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
    if wall_intersections is not None:
        assert row == wall_intersections.shape[0]
        n_wall_rays = wall_intersections.shape[1]
        wall_intersections = wall_intersections.astype(int)

    # Set up output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path_output_video, fourcc, fps, dimension)

    # fish point sizes
    if isinstance(fish_point_size, int):
        fish_point_size = [fish_point_size] * nnodes

    # Process video
    colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]
    for idx_frame in trange(row, desc="rendering video", colour="green"):
        frame = img.copy()
        for f in range(nfish):
            points = []
            for n in range(nnodes // 2):
                x = int(tracks[idx_frame, f * nnodes + 2 * n])
                y = int(tracks[idx_frame, f * nnodes + 2 * n + 1])
                points.append((x, y))
            for p, p_size in zip(points, fish_point_size):
                frame = cv2.circle(frame, p, p_size, colors[f], -1)
            # Lines between points
            if skeleton is not None:
                for p1, p2 in skeleton:
                    frame = cv2.line(
                        frame, points[p1], points[p2], colors[f], 1
                    )
            # raycasts
            if wall_intersections is not None:
                for wall_ray in range(n_wall_rays):
                    frame = cv2.circle(
                        frame,
                        wall_intersections[idx_frame, wall_ray],
                        2,
                        (0, 0, 255),
                        -1,
                    )
                    if wall_distances[idx_frame, wall_ray] < config["max_view"]:
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)
                    frame = cv2.line(
                        frame,
                        wall_intersections[idx_frame, wall_ray],
                        points[1],
                        color,
                        1,
                    )
                    frame = cv2.ellipse(
                        frame,
                        points[1],
                        (90, 90),
                        angle=0,
                        startAngle=0,
                        endAngle=360,
                        color=(100, 100, 100),
                    )
        out.write(frame)
        if show_video_during_rendering:
            cv2.imshow("fishy fish fish", frame)
            cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cv2.destroyAllWindows()


def main():
    # addTracksOnVideo( diff2, diff2_out, tracks, showvid=True, skeleton=[(0,1), (0,2), (0,3), (1,2), (1,3), (2,4), (3,5), (2,6), (3,7), (6,8), (7,8), (8,9)] )
    import reader

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
    addTracksOnTank(
        "videos/goal_ch.mp4",
        tracks,
        show_video_during_rendering=False,
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


if __name__ == "__main__":
    main()
