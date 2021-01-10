# File containing relevant functions for the nLocomotion Data structure
# The nLocomotion captures locomotion from one frame to another
# It is a np.array with following dimenstions: ( nfish, nframes, 3, nnodes )
# For the center node, the distance relative to the centerpoint of the previos frame and
# angle relative to the line between center and head point (orientation vector) of the
# previos frame is saved.
# For the head node, the distance to the new center is saved, and the angle of the vector
# between the new orientation vector and the old orientation vector is saved.
# For all other nodes, the distance to the new center point and the angle relative to the
# line between the new center and new head point is saved.
# An angle is saved as x and y coordinate on the unit circle.
# Due to that, the last 2 dimensions are organized as follows:
# [:,:,0] == Distances
# [:,:,1] == Angle, x coordinate on unit circle
# [:,:,2] == Angle, y coordinate on unit circle

import numpy as np
from numpy.core.function_base import linspace
from functions import getAngles, vectorLength, vectorsUnitAngle

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

        # Angle between orientation vector current and orientation vector new
        out[f,:,1,0], out[f,:,2,0] = vectorsUnitAngle( x_orivec_curr, y_orivec_curr, x_orivec_next, y_orivec_next )

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


def coordsFromLoc( oldCoords, locomotion ):
    """
    Computes new coordinates from oldCoords and locomotion

    Parameters
    ----------
    oldCoords : ndarray
        array containing x and y values for coordinates of a single time step:
        (nfish, 2, nnodes)
    locomotion : ndarray
        array containing locomotion for single time step: (nfish, 3, nnodes)

    Returns
    -------
    newCoords : ndarray
        array conating x and y values for coordinates when moving from oldCoords
        with locomotion: (nfish, 2, nnodes)
    """
    _, _, nnodes = oldCoords.shape
    # no assertions because this gets called a thousand times
    out = np.empty( oldCoords.shape )
    ## 1. Compute new center position
    x_head = oldCoords[:,0,0]
    y_head = oldCoords[:,1,0]
    x_center = oldCoords[:,0,1]
    y_center = oldCoords[:,1,1]

    x_orivec = x_head - x_center
    y_orivec = y_head - y_center

    # Compute angle to allocentric view (1,0)
    ang_orivec = getAngles( 1, 0, x_orivec, y_orivec )
    # print( x_orivec, y_orivec )
    # print( ang_orivec )
    # print( np.arctan( y_orivec / x_orivec ) )

    # Get relative angle of movement
    ang_relMov = getAngles( 1, 0, locomotion[:,1,1], locomotion[:,2,1] )

    # Add angles together and compute new center location
    ang_toNewCenter = ang_relMov + ang_orivec

    out[:,0,1] = np.cos( ang_toNewCenter ) * locomotion[:,0,1] + x_center
    out[:,1,1] = np.sin( ang_toNewCenter ) * locomotion[:,0,1] + y_center

    # New Head location
    ang_relOri = getAngles( 1, 0, locomotion[:,1,0], locomotion[:,2,0] )

    # Add angles together and compute head location
    ang_newOri = ang_relOri + ang_orivec

    #                                                       new center location
    out[:,0,0] = np.cos( ang_newOri ) * locomotion[:,0,0] + out[:,0,1]
    out[:,1,0] = np.sin( ang_newOri ) * locomotion[:,0,0] + out[:,1,1]

    # Iterate through rest of nodes
    for n in range( 2, nnodes ):
        # get angle relative to center node
        ang_relNode = getAngles( 1, 0, locomotion[:,1,n], locomotion[:,2,n] )
        ang_node = ang_relNode + ang_newOri
        out[:,0,n] = np.cos( ang_node ) * locomotion[:,0,n] + out[:,0,1]
        out[:,1,n] = np.sin( ang_node ) * locomotion[:,0,n] + out[:,1,1]

    return out


def nLoc2trackData( nLocomotion, startCoordinates ):
    """
    Converts nLocomotion to trackData given start coordinates

    Parameters
    ----------
    nLocomotion : ndarray
        array containing movement data between frames in form of (nfish, nframes - 1, 3, nnodes)
    startCoordinates : ndarray
        array containing start point coordinates in form of (nfish, 2, nnodes),
        whereas [nfish, 0, nnodes] are the x values and [nfish, 1, nnodes] are the y values

    Returns
    -------
    trackData : ndarray
        Coordinates in form of (nfish, nframes, 2, nnodes)
    """
    # Input check and assertions
    nfish, nframes, ncoords, nnodes = nLocomotion.shape
    nframes += 1
    assert ncoords == 3, "nLocomotion needs to be 3 in second to last dimension!"
    nfish2, ncoords2, nnodes2 = startCoordinates.shape
    assert ncoords2 == 2, "startCoordinates need to have x and y values seperate"
    assert nfish == nfish2, "Different amount of fish for nLocomotion and startCoordinates"
    assert nnodes == nnodes2, "Different amount of nodes for nLocomotion and startCoordinates"

    # Create Output
    out = np.empty( (nfish, nframes, 2, nnodes) )

    # Set startpoints
    out[:,0] = startCoordinates

    # Iterate and compute coordinates
    for i in range( nframes - 1 ):
        out[:,i+1] = coordsFromLoc( out[:,i], nLocomotion[:,i] )

    return out


def nLoc2binnednLoc( nLocomotion, nbins=40 ):
    """
    Converts nLocomotion to binned nLocomotion, that means
    every entry is converted into a oneHotVector encoding
    where each value represents the probability of an action
    being in a certain bin.

    Parameter
    ---------
    nLocomotion : ndarray
        array containing movement data between frames in form of (nfish, nframes - 1, 3, nnodes)
    bins : int
        Number specifying how many bins are used

    Returns
    -------
    binnednLoc : ndarray
        array containing movement data in a binned manner in form of
        (nfish, nframes - 1, 3, nnodes, nbins)
    """
    nfish, nframes, ncoords, nnodes = nLocomotion.shape

    out = np.empty( ( nfish, nframes, ncoords, nnodes, nbins ) )

    # Since every Node has different average values and value ranges, bins are made separately
    nodeDistanceBins = [
        np.linspace( 0, 40, nbins - 1 ),        # Head
        np.linspace( 0, 20, nbins - 1 ),        # Movement distance
        np.linspace( 0, 15, nbins - 1 ),        # Left Base
        np.linspace( 0, 15, nbins - 1 ),        # Right Base
        np.linspace( 0, 35, nbins - 1 ),        # Left Fin
        np.linspace( 0, 35, nbins - 1 ),        # Right Fin
        np.linspace( 5, 35, nbins - 1 ),        # Left Body
        np.linspace( 5, 35, nbins - 1 ),        # Right Body
        np.linspace( 5, 65, nbins - 1 ),        # Tail
        np.linspace( 10, 70, nbins - 1 ),       # Tail Fin
    ]

    angbins = linspace( -1, 1, nbins -1 )

    for n in range( nnodes ):
        idx_c = np.digitize( nLocomotion[:,:,0,n], nodeDistanceBins[n] )
        out[:,:,0,n] = np.eye( nbins )[idx_c]

    idx_c = np.digitize( nLocomotion[:,:,1,:], angbins )
    out[:,:,1,:] = np.eye( nbins )[idx_c]

    idx_c = np.digitize( nLocomotion[:,:,2,:], angbins )
    out[:,:,2,:] = np.eye( nbins )[idx_c]

    return out


def main():
    import data_io
    trackData = data_io.lazyTrackData( 1 )
    # print( trackData.shape )
    # print( trackData )
    nloc = trackData2nLoc( trackData )
    # print( nloc.shape )
    # print( nloc )
    # print( " " )
    f = 2
    # print( "avg:", np.mean( nloc[f,:,0], axis=0 ) )
    # print( "max:", np.amax( nloc[f,:,0], axis=0 ) )
    # print( "min:", np.amin( nloc[f,:,0], axis=0 ) )
    # startpos =  trackData[:,0]
    # newTrackData = nLoc2trackData( nloc, startpos )
    # print( newTrackData.shape )
    # print( newTrackData )
    # nbins = 40
    # bins = np.linspace( -20, 20, nbins )
    # print( bins )
    # test = np.array( [ np.array( [0.5,10,3.1,4.2] ), np.array( [-15,19.8,3,4.2] ), np.array( [20.3,2.1,9.3,-2] ) ] )
    # print( np.digitize( test, bins ) )
    nloc[f,100,0,1] = 18
    bLoc = nLoc2binnednLoc( nloc )
    # np.set_printoptions(threshold=np.inf)
    # print( nloc[f,100,:,1] )
    # print( bLoc[f,100,:,1] )

    pass

if __name__ == "__main__":
    main()