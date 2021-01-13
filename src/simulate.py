# Function to simulate fish behavior with a trained model


import numpy as np
from numpy.testing._private.utils import break_cycles
import tensorflow as tf
from data_io import lazyRaycastData, lazyTrackData
from functions import defineLines, getAngle, getRedPoints, vectorLength, vectorsUnitAngle
from nLocomotion import getNodeAngleBinValue, getNodeDistanceBinValues, getNodeDistanceBins, nLoc2binnednLoc, trackData2nLoc
from raycasts import Raycast
from train import getnView, lazyNodeIndices, prepare_data
from visualization import addTracksOnTank


def loadStartData( nodes, hist_size, target_size, nbins ):
    """
    Returns a startpoint

    Parameter
    ---------
    nodes : array
        array containing indices of nodes used in tracks
    hist_size : int
        amount of frames taken into account to make a
        prediction
    target_size : int
        amount of frames predicted in one prediction
    nbins : int
        Number deciding in how many bins the locomotion
        should be discretized in

    Returns
    -------
    (x_train, y_train, x_val, y_val) : (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Tuple containing training and validation datasets
    """
    TRACK = 1


    trackData = lazyTrackData( TRACK )[:,:hist_size + 2]
    wRays = lazyRaycastData( TRACK )[:,:hist_size + 2]

    trackData = trackData[:,:,:,nodes]

    bLoc = nLoc2binnednLoc( trackData2nLoc( trackData ), nbins )

    nfish, nframes, ncoords, nnodes = trackData.shape
    nfish2, nframes2, nwRays = wRays.shape

    assert ncoords == 2
    assert nfish == nfish2
    assert nframes == nframes2


    N_NVIEW = ( nfish - 1 ) * nnodes * 3
    N_WRAYS = nwRays
    N_NLOC = nnodes * 3 * nbins
    ndataPoints = N_NVIEW + N_WRAYS + N_NLOC

    out = []
    idxs = list( range( nfish ) )
    for f in range( nfish ):
        fdataset = np.empty( ( nframes - 1, ndataPoints ) )

        # nView
        idxOther = [ x for x in idxs if x != f ]
        fnView = getnView( trackData[f], trackData[idxOther] )
        fdataset[:,:N_NVIEW] = np.reshape( np.swapaxes( fnView[:,1:], 0, 1 ), ( nframes - 1, N_NVIEW ) )
        # print( fnView[:,1] )
        # print( fdataset[69,:N_NVIEW] )
        # print( trackData[f,70] )
        # print( fnView[:,70] )
        # print( getnViewSingle( trackData[f,70], trackData[idxOther,70] ) )
        # RayCasts
        fdataset[:,N_NVIEW:-N_NLOC] = wRays[f,1:]

        # Locomotion
        fbLoc = np.reshape( bLoc[f], ( nframes - 1, N_NLOC ) )
        fdataset[:,-N_NLOC:] = fbLoc

        # print( fnLoc[0] )
        # print( nLoc[f][0] )
        # import sys
        # sys.exit()
        # Target
        ftarget = np.empty( ( nframes - 1, N_NLOC ) )
        ftarget = fbLoc

        # Prepare data
        x_train, _ = prepare_data( fdataset, ftarget, 0, None, hist_size, target_size, 1, single_step=True )
        out.append( ( x_train[0], trackData[f,hist_size], ) )

    return out


def getnViewSingle( tracksCurr, tracksOther ):
    """
    Returns distance and angle from current fish to all nodes of other fish

    Parameter
    ---------
    tracksCurr : np.array
        trackData for fish for which the nView is computed
        shape: (2, nnodes)
    tracksOther : np.array
        trackData for other fish the current fish is looking at
        shape: (nOtherFish, 2, nnodes)

    Returns
    -------
    nView : np.array
        Distance and angle to all nodes of other fish from center node
        in current fish.
        [:,:,0,:] = distance; [:,:,1/2,:] = x/y coordinate on unit circle for angle
        shape: (nOtherFish, 3, nnodes)
    """
    _, nnodes = tracksCurr.shape
    nOtherFish, _, _ = tracksOther.shape

    out = np.empty( ( nOtherFish, 3, nnodes ) )
    x_head_curr = tracksCurr[0,0]
    y_head_curr = tracksCurr[1,0]
    x_center_curr = tracksCurr[0,1]
    y_center_curr = tracksCurr[1,1]
    # head - center
    x_orivec_curr = x_head_curr - x_center_curr
    y_orivec_curr = y_head_curr - y_center_curr
    # 2. compute values for every fish and node
    for f in range( nOtherFish ):
        # Iterate through nodes
        for n in range( nnodes ):
            # Get Node
            x_node_other = tracksOther[f,0,n]
            y_node_other = tracksOther[f,1,n]
            # Vector between current center and Node (node - center)
            x_centernode_vec = x_node_other - x_center_curr
            y_centernode_vec = y_node_other - y_center_curr
            # Save in output
            out[f,0,n] = vectorLength( x_centernode_vec, y_centernode_vec )
            out[f,1,n], out[f,2,n] = vectorsUnitAngle( x_orivec_curr, y_orivec_curr, x_centernode_vec, y_centernode_vec )

    return out


def softmax( x ):
    """
    Computes softmax
    """
    # xrel = x - np.max( x )
    return np.exp( x ) / np.sum( np.exp( x ), axis=0 )


def selectAction( pred ):
    roll = np.random.rand() - pred[0]
    idx = 0
    # while roll > 0:
    #     idx+= 1
    #     roll-= pred[idx]
    # return idx
    return np.argmax( pred )


def simulate( model, steps, startData, nnodes, n_wrays, fov_walls, max_view_range ):
    """
    """
    nfish = len( startData )
    nbins = 40

    N_NVIEW = ( nfish - 1 ) * nnodes * 3
    N_WRAYS = n_wrays
    N_NLOC = nnodes * 3 * nbins
    ndataPoints = N_NVIEW + N_WRAYS + N_NLOC

    modelinputs = np.asarray( [ x[0] for x in startData ] )
    trackData = np.empty( ( nfish, steps + 1, 2, nnodes ) )
    angle_ori = np.empty( ( nfish ) ) # Angle Fish has to allocentric view
    prevLoc = np.empty( ( nfish, N_NLOC ) )

    # Set beginning positions
    for f in range( nfish ):
        trackData[f,0] = startData[f][1]

    # Compute first angle
    for f in range( nfish ):
        x_head = trackData[f,0,0,0]
        y_head = trackData[f,0,1,0]
        x_center = trackData[f,0,0,1]
        y_center = trackData[f,0,1,1]
        x_cH_vec = x_head - x_center
        y_cH_vec = y_head - y_center
        angle_ori[f] = getAngle( ( 1, 0 ), ( x_cH_vec, y_cH_vec ), "radians" )

    # Handle Raycasts
    walllines = defineLines( getRedPoints( path = "data/final_redpoint_wall.jpg" ) )
    unnecessary = 10
    raycast_object = Raycast( walllines, unnecessary, n_wrays, unnecessary, fov_walls, max_view_range, nfish )

    # Get bin values
    nodeDistanceBinValues = getNodeDistanceBinValues( getNodeDistanceBins( nbins ) )
    nodeAngleBinValues = getNodeAngleBinValue( np.linspace( -1, 1, nbins - 1 ) )

    idxs = list( range( nfish ) )
    # Main Loop
    for i in range( steps ):
        if i % 500 == 0:
            print( f"Frame {i:6}" )

        for f in range( nfish ):
            # 1. Compute input for fish
            # 2. Compute prediction
            # 3. Compute new positions

            # 1. Input for fish model
            if i != 0:
                singleFrame = np.empty( ndataPoints )

                # nView
                idxOther = [ x for x in idxs if x != f ]
                singleFrame[:N_NVIEW] = getnViewSingle( trackData[f,i], trackData[idxOther,i] ).reshape( N_NVIEW )
                # print( singleFrame[:N_NVIEW] )
                # print( modelinputs[f,69,:N_NVIEW] )

                # wRays
                center = np.asarray( [ trackData[f,i,0,1], trackData[f,i,1,1] ] )
                head = np.asarray( [ trackData[f,i,0,0], trackData[f,i,1,0] ] )
                vec_ch = head - center
                singleFrame[N_NVIEW:-N_NLOC] = raycast_object._getWallRays( center, vec_ch )

                # print( singleFrame[N_NVIEW:-N_NLOC] )
                # print( modelinputs[f,69,N_NVIEW:-N_NLOC] )

                # Locomotion from previous step
                singleFrame[-N_NLOC:] = prevLoc[f]

                # Add frame to modelinput stack
                modelinputs[f,:-1] = modelinputs[f,1:]
                modelinputs[f,-1] = singleFrame

            # 2. Compute Prediction
            rawprediction = model.predict( np.array( [ modelinputs[f] ] ) )[0]

            predidx = np.empty( nnodes * 3, dtype=int )
            for n in range( 3 * nnodes):
                # Run softmax
                prevLoc[f][n * nbins:(n + 1) * nbins] = softmax( rawprediction[n * nbins:(n + 1) * nbins] )

                # Convert predictions into indices
                predidx[n] = selectAction( prevLoc[f][n * nbins:(n + 1) * nbins] )

            predvals = np.empty( nnodes * 3 )
            # Distance nodes
            for n in range( nnodes ):
                predvals[n] = nodeDistanceBinValues[n][predidx[n]]
            # Angles
            for n in range( nnodes ):
                predvals[nnodes + n] = nodeAngleBinValues[predidx[nnodes + n]]
                predvals[2 * nnodes + n] = nodeAngleBinValues[predidx[2 * nnodes + n]]

            print( f"S{i}:f{f}: {predvals}")
            # 3. Compute new fish position
            # a. new center position
            x_center_curr = trackData[f,i,0,1]
            y_center_curr = trackData[f,i,1,1]

            angle_relMov = getAngle( ( 1, 0 ), ( predvals[nnodes + 1], predvals[2 * nnodes + 1] ), "radians" )

            angle_toNewCenter = angle_ori[f] + angle_relMov

            trackData[f,i + 1,0,1] = np.cos( angle_toNewCenter ) * predvals[1] + x_center_curr
            trackData[f,i + 1,1,1] = np.sin( angle_toNewCenter ) * predvals[1] + y_center_curr

            # b. new head position
            angle_relOri = getAngle( ( 1, 0 ), ( predvals[nnodes], predvals[2 * nnodes] ), "radians" )

            angle_ori[f] = angle_ori[f] + angle_relOri

            #                                                               new center location
            trackData[f,i + 1,0,0] = np.cos( angle_ori[f] ) * predvals[0] + trackData[f,i + 1,0,1]
            trackData[f,i + 1,1,0] = np.sin( angle_ori[f] ) * predvals[0] + trackData[f,i + 1,1,1]

            # c. Iterate through rest of nodes
            for n in range( 2, nnodes ):
                angle_relNode = getAngle( (1, 0 ), ( predvals[nnodes + n], predvals[nnodes * 2 + n] ), "radians" )
                angle_node = angle_relNode + angle_ori[f]
                trackData[f,i + 1,0,0] = np.cos( angle_node ) * predvals[0] + trackData[f,i + 1,0,1]
                trackData[f,i + 1,1,0] = np.sin( angle_node ) * predvals[0] + trackData[f,i + 1,1,1]

    return trackData


def main():
    HIST_SIZE = 70
    TARGET_SIZE = 0
    BATCH_SIZE = 10
    BUFFER_SIZE = 10000
    EPOCHS = 1
    SPLIT = 0.9
    N_WRAYS = 15
    FOV_WALLS = 180
    MAX_VIEW_RANGE = 709
    U_LSTM = 50
    U_DENSE = 50
    N_BINS = 40

    tracks = [1]
    nodes = lazyNodeIndices( 2 )
    startData = loadStartData( nodes, HIST_SIZE, TARGET_SIZE, N_BINS )
    # print( startData )
    # print( startData[0][0].shape, startData[0][1].shape )
    modelname = "new_2n_900_600_10b_70h"
    model = tf.keras.models.load_model( "models/" + modelname )
    trackData = simulate( model, 100, startData, len( nodes ), N_WRAYS, FOV_WALLS, MAX_VIEW_RANGE )
    testvid1 = 'data/videos/testvid1.avi'
    addTracksOnTank( testvid1, trackData, showvid=True, skeleton=[(0,1)] )
    return

if __name__ == "__main__":
    main()
