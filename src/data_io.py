# File to load data

# trackData is saved in following shape:
# ( nfish, nframes, 2, nnodes )
# The last 2 dimensions are the x and y coordinates.

import numpy as np
import pandas as pd


def loadTrackData( path ):
    """
    Loads trackData from path

    Parameters
    ----------
    path : path, filename
        Path to dataSet

    Returns
    -------
    trackData : np.array
        x and y coordinates of fishes in following
        dimension: (nfish, nframes, 2, nnodes)
    """
    return np.load( path )


def lazyTrackData( n ):
    """
    Loads predefined datasets

    Parameters
    ----------
    n : int
        Number of dataset to be loaded in, whereas:
        1:diff1, 2:diff2, 3:diff3, 4:diff4,
        5:same1, 6:same3, 7:same4, 8:same5

    Returns
    -------
    trackData : np.array
        x and y coordinates of fishes in following
        dimension: (nfish, nframes, 2, nnodes)
    """
    assert n >= 1 and n <= 8, "n needs to be between 1 and 8!"
    track1 = "data/trackData/track1.npy"
    track2 = "data/trackData/track2.npy"
    track3 = "data/trackData/track3.npy"
    track4 = "data/trackData/track4.npy"
    track5 = "data/trackData/track5.npy"
    track6 = "data/trackData/track6.npy"
    track7 = "data/trackData/track7.npy"
    track8 = "data/trackData/track8.npy"
    tracks = [ track1, track2, track3, track4, track5, track6, track7, track8 ]
    return np.load( tracks[n - 1] )


def loadRaycastData( path ):
    """
    Loads raycastData from path, only containing wall Rays

    Parameters
    ----------
    path : path, filename
        Path to dataSet

    Returns
    -------
    raycastData : np.array
        loads agent bins and wall rays.
        Result has dimension: (nfish, nframes, nWallRays)
    """
    return np.load( path )[:,:,-15:]


def lazyRaycastData( n ):
    """
    Loads raycastData by number, only containing wall Rays

    Parameters
    ----------
    n : int
        Number of dataset to be loaded in
        1:diff1, 2:diff2, 3:diff3, 4:diff4,
        5:same1, 6:same3, 7:same4, 8:same5

    Returns
    -------
    raycastData : np.array
        loads agent bins and wall rays.
        Result has dimension: (nfish, nframes, nWallRays)
    """
    assert n >= 1 and n <= 8, "n needs to be between 1 and 8!"
    ray1 = "data/raycastData/raycast1.npy"
    ray2 = "data/raycastData/raycast2.npy"
    ray3 = "data/raycastData/raycast3.npy"
    ray4 = "data/raycastData/raycast4.npy"
    ray5 = "data/raycastData/raycast5.npy"
    ray6 = "data/raycastData/raycast6.npy"
    ray7 = "data/raycastData/raycast7.npy"
    ray8 = "data/raycastData/raycast8.npy"
    rays = [ ray1, ray2, ray3, ray4, ray5, ray6, ray7, ray8 ]
    return np.load( rays[n - 1] )[:,:,-15:]


def loadMean( file="data/mean.npy" ):
    """
    Loads the mean for the data used to train the nModel

    Parameter
    ---------
    file : filename/path
        file including the mean for training and target
        data

    Returns
    -------
    means : tuple of np.ndarray
        mean, std, mean target, std target
    """
    return np.load( file, allow_pickle=True )


def saveModel( model, modelname ):
    """
    Saves the tf model in given path

    Parameter
    ---------
    model : tf.keras.models
        A tf keras model
    path : string/path
        Path to directory where model should be saved to
    """
    path = "models/" + modelname
    if not os.path.isdir( path ):
        os.mkdir( path )
    return model.save( path )
