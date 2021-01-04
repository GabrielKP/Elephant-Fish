# File to handle network training
from argparse import ArgumentParser
import numpy as np
from data_io import lazyRaycastData, lazyTrackData, loadMean, loadRaycastData, loadTrackData
from nLocomotion import trackData2nLoc
from functions import vectorLength, vectorsUnitAngle


def lazyNodeIndices( x ):
    """
    Returns node indices depending on x
    """
    assert x % 2 == 0 and x != 0 and x <= 10, "x needs to be 2,4,6,8 or 10"
    n2 = [0,1]
    n4 = [0,1,8,9]
    n6 = [0,1,2,3,8,9]
    n8 = [0,1,2,3,6,7,8,9]
    n10 = [0,1,2,3,4,5,6,7,8,9]
    ns = ( n2, n4, n6, n8, n10, )
    return ns[np.argmax( np.array( [2,4,6,8,10] ) == x )]


def getnView( tracksCurr, tracksOther ):
    """
    Returns distance and angle from current fish to all nodes of other fish

    Parameter
    ---------
    tracksCurr : np.array
        trackData for fish for which the nView is computed
        shape: (nframes, 2, nnodes)
    tracksOther : np.array
        trackData for other fish the current fish is looking at
        shape: (nOtherFish, nframes, nnodes)

    Returns
    -------
    nView : np.array
        Distance and angle to all nodes of other fish from center node
        in current fish.
        [:,:,0,:] = distance; [:,:,1/2,:] = x/y coordinate on unit circle for angle
        shape: (nOtherFish, nframes, 3, nnodes)
    """
    nframes, ncoords, nnodes = tracksCurr.shape
    nOtherFish, nframes2, ncoords2, nnodes2 = tracksOther.shape
    assert nframes == nframes2
    assert ncoords == ncoords2
    assert ncoords == 2
    assert nnodes == nnodes2

    out = np.empty( ( nOtherFish, nframes, 3, nnodes ) )
    x_head_curr = tracksCurr[:,0,0]
    y_head_curr = tracksCurr[:,1,0]
    x_center_curr = tracksCurr[:,0,1]
    y_center_curr = tracksCurr[:,1,1]
    # head - center
    x_orivec_curr = x_head_curr - x_center_curr
    y_orivec_curr = y_head_curr - y_center_curr
    # 2. compute values for every fish and node
    for f in range( nOtherFish ):
        # Iterate through nodes
        for n in range( nnodes ):
            # Get Node
            x_node_other = tracksOther[f,:,0,n]
            y_node_other = tracksOther[f,:,1,n]
            # Vector between current center and Node (node - center)
            x_centernode_vec = x_node_other - x_center_curr
            y_centernode_vec = y_node_other - y_center_curr
            # Save in output
            out[f,:,0,n] = vectorLength( x_centernode_vec, y_centernode_vec )
            out[f,:,1,n], out[f,:,2,n] = vectorsUnitAngle( x_orivec_curr, y_orivec_curr, x_centernode_vec, y_centernode_vec )

    return out


def createMean( file, verbose=0 ):
    """
    Creates the mean and standart variation from all data points
    and saves it in file as .npy.

    Parameter
    ---------
    file : string/path
        filename mean should be saved to.
    verbose : int
        0 no console output
        1 or higher outputs std and mean on console
    """

    totaldataset = []
    targetdataset = []
    for i in range( 2, 9 ):
        trackData = lazyTrackData( i )
        wRays = lazyRaycastData( i )
        nLoc = trackData2nLoc( trackData )

        nfish, nframes, ncoords, nnodes = trackData.shape
        nfish2, nframes2, nwRays = wRays.shape
        nfish3, nframes3, ncoords3, nnodes3 = nLoc.shape
        assert ncoords == 2
        assert ncoords3 == ncoords + 1
        assert nfish == nfish2
        assert nfish3 == nfish2
        assert nframes == nframes2
        assert nframes  == nframes3 + 1
        assert nnodes == nnodes3

        N_NVIEW = ( nfish - 1 ) * nnodes * 3
        N_WRAYS = nwRays
        N_NLOC = nnodes * 3

        # Number of Datapoints in every row:
        # nnView for fish (dis + angle to every other fishhnode)
        # == nnodes * 3 * (nfish - 1)
        # nwRays for fish ( 15 )
        # nnLoc for fish ( 3 * nnodes )
        ndataPoints = N_NVIEW + N_WRAYS + N_NLOC

        idxs = list( range( nfish ) )

        for f in range( nfish ):
            fdataset = np.empty( ( nframes - 1, ndataPoints ) )

            # nView
            idxOther = [ x for x in idxs if x != f ]
            fnView = getnView( trackData[f], trackData[idxOther] )
            fdataset[:,:N_NVIEW] = np.reshape( fnView[:,1:], ( nframes - 1, N_NVIEW ) )
            # RayCasts
            fdataset[:,N_NVIEW:-N_NLOC] = wRays[f,1:]
            # Locomotion
            fnLoc = np.reshape( nLoc[f], ( nframes - 1, N_NLOC ) )
            fdataset[:,-N_NLOC:] = fnLoc

            # Target
            ftarget = np.empty( ( nframes - 1, N_NLOC ) )
            ftarget = fnLoc

            # for mean and std calculation
            totaldataset.append( np.array( fdataset ) )
            targetdataset.append( np.array( ftarget ) )


    totaldataset = np.concatenate( totaldataset, axis=0 )
    meanv = totaldataset.mean( axis=0 )
    std = totaldataset.std( axis=0 )

    targetdataset = np.concatenate( targetdataset, axis=0 )
    meanTGT = targetdataset.mean( axis=0 )
    stdTGT = targetdataset.std( axis=0 )

    if verbose:
        print( "mean   :" )
        print( meanv )
        print( "std    :" )
        print( std )
        print( "meanTGT:" )
        print( meanTGT )
        print( "stTGT  :" )
        print( stdTGT )

    np.save( file, np.array( [meanv, std, meanTGT, stdTGT], dtype=object ) )
    print( f"Written to File: {file}" )


def prepare_data( dataset, target, start_index, end_index, history_size, target_size, step, single_step=False ):
    """
    Function to create datasets with history_size
    and according labels with target_size
    Originally taken from
    https://www.tensorflow.org/tutorials/structured_data/time_series

    Parameter
    ---------
    @todo

    Returns
    -------
    @todo

    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len( dataset ) - target_size

    for i in range( start_index, end_index ):
        indices = range( i - history_size, i, step )
        data.append( dataset[indices] )

        if single_step:
            labels.append( target[i + target_size] )
        else:
            labels.append( target[i:i + target_size] )

    return np.array( data ), np.array( labels )


def loadData( tracks, split, hist_size, target_size, meanFile="data/mean.npy", files=False, wallRayFiles=None ):
    """
    Loads Data specified in tracks into train and val data

    Parameter
    ---------
    tracks : array
        array containing integers of tracksets [1,8],
        when paths is set to True the array needs to
        contain file names to tracksets
    split : float (0,1)
        number between 0 and 1 specifying split of
        test and validation data
    hist_size : int
        amount of frames taken into account to make a
        prediction
    target_size : int
        amount of frames predicted in one prediction
    meanFile : file/pathToFile
        File containing mean and standard variation for
        data given
    files : bool
        switches tracks statement to work with filenames
        instead of integers. If true, wallRays will need
        to be given as well
    wallRayFiles : array
        Array of filenames containing wallRays in matching
        order to the tracks given in first statement.
        Only needed if files is set to true.

    Returns
    -------
    (x_train, y_train, x_val, y_val) : (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Tuple containing training and validation datasets
    """
    assert not files or wallRayFiles is not None, "wallRay Files need to be given if files is set to true"

    # Load mean
    mean, std, meanTGT, stdTGT = loadMean( meanFile )

    x_data_train = []
    y_data_train = []
    x_data_val = []
    y_data_val = []
    for i, track in enumerate( tracks ):
        if files:
            trackData = loadTrackData( track )
            wRays = loadRaycastData( wallRayFiles[i] )
        else:
            trackData = lazyTrackData( track )
            wRays = lazyRaycastData( track )

        nLoc = trackData2nLoc( trackData )

        nfish, nframes, ncoords, nnodes = trackData.shape
        nfish2, nframes2, nwRays = wRays.shape
        nfish3, nframes3, ncoords3, nnodes3 = nLoc.shape
        print( nframes )
        assert ncoords == 2
        assert ncoords3 == ncoords + 1
        assert nfish == nfish2
        assert nfish3 == nfish2
        assert nframes == nframes2
        assert nframes  == nframes3 + 1
        assert nnodes == nnodes3

        N_NVIEW = ( nfish - 1 ) * nnodes * 3
        N_WRAYS = nwRays
        N_NLOC = nnodes * 3
        ndataPoints = N_NVIEW + N_WRAYS + N_NLOC

        idx_split = int( ( nframes - 1 ) * split )

        idxs = list( range( nfish ) )
        for f in range( nfish ):
            fdataset = np.empty( ( nframes - 1, ndataPoints ) )

            # nView
            idxOther = [ x for x in idxs if x != f ]
            fnView = getnView( trackData[f], trackData[idxOther] )
            fdataset[:,:N_NVIEW] = np.reshape( fnView[:,1:], ( nframes - 1, N_NVIEW ) )
            # RayCasts
            fdataset[:,N_NVIEW:-N_NLOC] = wRays[f,1:]
            # Locomotion
            fnLoc = np.reshape( nLoc[f], ( nframes - 1, N_NLOC ) )
            fdataset[:,-N_NLOC:] = fnLoc

            # Target
            ftarget = np.empty( ( nframes - 1, N_NLOC ) )
            ftarget = fnLoc

            # Mean and std
            fdataset = np.nan_to_num( ( fdataset - mean ) / std )
            ftarget = np.nan_to_num( ( ftarget - meanTGT ) / stdTGT )

            # Prepare data
            x_train, y_train = prepare_data( fdataset, ftarget, 0, idx_split, hist_size, target_size, 1, single_step=True )
            x_data_train.append( x_train )
            y_data_train.append( y_train )

            x_val, y_val = prepare_data( fdataset, ftarget, idx_split, None, hist_size, target_size, 1, single_step=True )
            x_data_val.append( x_val )
            y_data_val.append( y_val )

    x_data_train = np.concatenate( x_data_train, axis=0 )
    y_data_train = np.concatenate( y_data_train, axis=0 )
    x_data_val = np.concatenate( x_data_val, axis=0 )
    y_data_val = np.concatenate( y_data_val, axis=0 )

    return x_data_train, y_data_train, x_data_val, y_data_val


def train(
    trackdata,
    nfish,
    nodes,
    modeltag,
    units_lstm,
    units_dense,
    epochs,
    split,
    batch_size,
    buffer_size,
    hist_size,
    target_size,
    fov_walls,
    max_view_range
):
    pass


def main():
    # Parse Arguments
    print( quickNodes( 8 ) )
    return
    parser = ArgumentParser()

    parser.add_argument( "track",
        help="file includings tracks on which network should be trained on" )
    parser.add_argument( "--name",
        help="model Name" )
    pass

if __name__ == "__main__":
    main()
