# File containing relevant functions for the nLocomotion Data structure
# The nLocomotion captures locomotion from one frame to another
# It is a np.array with following dimenstions: ( nfish, nframes, 3, nnodes )
# For the center node, the distance relative to the centerpoint of the previos frame and
# angle relative to the line between center and head point (orientation vector) of the
# previos frame is saved.
# For all other nodes, the distance to the new center point and the angle relative to the
# line between the new center and new head point is saved. Thus, the head node always has
# an angle of 0.
# An angle is saved as x and y coordinate on the unit circle.
# Due to that, the last 2 dimensions are organized as follows:
# [:,:,0] == Distances
# [:,:,1] == Angle, x coordinate on unit circle
# [:,:,2] == Angle, y coordinate on unit circle

import data_io
import numpy as np
from functions import vectorLength, vectorsUnitAngle

def trackData2nLoc( trackData ):
    """
    Converts trackData into nLocomotion

    Parameters
    ----------
    trackData : ndarray
        Coordinates in form of (nfish, nframes, 2, nnodes)

    Returns
    -------
    nLoc : ndarray
        Locomotion between frames of trackData in form of
        (nfish, nframes - 1, 3, nnodes)
    """
    nfish, nframes, ncords, nnodes = trackData.shape
    assert ncords == 2, "Function only accepts x and y coordinates and not more or less"
    assert nframes >= 2, "At least 2 frames are needed"
    assert nnodes >= 2, "Center and Head Node are always needed"

    # create output structure
    out = np.empty( ( nfish, nframes - 1, 3, nnodes ) )

    # iterate through every fish in vectorized manner
    for f in range(nfish):
        ## 1. Set distances and angles for head and center node

        # orientation vector between next head and center
        x_head_next = trackData[f,1:,0,0]
        y_head_next = trackData[f,1:,1,0]
        x_center_next = trackData[f,1:,0,1]
        y_center_next = trackData[f,1:,1,1]
        # head - center
        x_orivec_next = x_head_next - x_center_next
        y_orivec_next = y_head_next - y_center_next

        # orientation vector current head and center
        x_head_curr = trackData[f,:-1,0,0]
        y_head_curr = trackData[f,:-1,1,0]
        x_center_curr = trackData[f,:-1,0,1]
        y_center_curr = trackData[f,:-1,1,1]
        # head - center
        x_orivec_curr = x_head_curr - x_center_curr
        y_orivec_curr = y_head_curr - y_center_curr

        ## 2. Distance between next and current center

        # center_next - center
        x_mov_vec = x_center_next - x_center_curr
        y_mov_vec = y_center_next - y_center_curr

        # Compute length and save as distance
        out[f,:,0,1] = vectorLength( x_mov_vec, y_mov_vec )

        ## 3. Angle of movement
        # angle of current look vector to movement vector
        out[f,:,1,1], out[f,:,2,1] = vectorsUnitAngle( x_orivec_curr, y_orivec_curr, x_mov_vec, y_mov_vec )

        ## 4. Distance and angle for head node
        out[f,:,0,0] = vectorLength( x_orivec_next, y_orivec_next )

        # Angle between head and center is always 0 (1,0)
        out[f,:,1,0], out[f,:,2,0] = 1.0, 0.0

        ## Set every other node in relation to orientation of center node
        for n in range( nnodes - 2 ):
            # 1. Distance
            x_node_next = trackData[f,1:,0,n + 2]
            y_node_next = trackData[f,1:,1,n + 2]
            x_centernode_vec = x_node_next - x_center_next
            y_centernode_vec = y_node_next - y_center_next
            out[f,:,0,n + 2] = vectorLength( x_centernode_vec, y_centernode_vec )
            # 2. Angle
            out[f,:,1,n + 2], out[f,:,2,n + 2] = vectorsUnitAngle( x_orivec_next, y_orivec_next, x_centernode_vec, y_centernode_vec )

    return out


def nLoc2trackData():
    pass

trackData = data_io.lazytrackData( 1 )[1:2]
print( trackData[0,0:2].shape )
print( trackData[0,0:2] )
nloc = trackData2nLoc( trackData )
print( nloc[0,0].shape )
print( nloc[0,0] )