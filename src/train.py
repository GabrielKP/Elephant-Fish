# File to handle network training
from argparse import ArgumentParser
import numpy as np

def quickNodeIndices( x ):
    """
    Returns node indices depending on m
    """
    assert x % 2 == 0 and x != 0 and x <= 10, "x needs to be 2,4,6,8 or 10"
    n2 = [0,1]
    n4 = [0,1,8,9]
    n6 = [0,1,2,3,8,9]
    n8 = [0,1,2,3,6,7,8,9]
    n10 = [0,1,2,3,4,5,6,7,8,9]
    ns = ( n2, n4, n6, n8, n10, )
    return ns[np.argmax( np.array( [2,4,6,8,10] ) == x )]

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