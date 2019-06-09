"""
    utils.py - Some convenience functions for using data from
        Joint 2D-3D-Semantic Data for Indoor Scene Understanding
           I. Armeni*, A. Sax*, A. Zamir, S. Savarese
        Website: 3dsemantics.stanford.edu
        Paper: https://arxiv.org/pdf/1702.01105.pdf 
        
    Code Author: Alexander Sax
    
    Usage: For import only. i.e. 'import utils.py'
      Dependencies include scipy, OpenEXR
"""

# import array
# import Imath
#
# import numpy
# import OpenEXR
import json
from   scipy.ndimage import imread
import numpy as np

""" Semantics """
def get_index( color ):
    ''' Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    '''
    return color[0] * 256 * 256 + color[1] * 256 + color[2]

def get_color( i ):
    ''' Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    '''
    b = ( i ) % 256  # least significant byte
    g = ( i >> 8 ) % 256
    r = ( i >> 16 ) % 256 # most significant byte 
    return r,g,b

""" Label functions """
def load_labels( label_file ):
    """ Convenience function for loading JSON labels """
    with open( label_file ) as f:
        return json.load( f )

def parse_label( label ):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split( "_" )
    res[ 'instance_class' ] = clazz
    res[ 'instance_num' ] = int( instance_num )
    res[ 'room_type' ] = room_type
    res[ 'room_num' ] = int( room_num )
    res[ 'area_num' ] = int( area_num )
    return res


""" EXR Functions """
def normalize_array_for_matplotlib( arr_to_rescale ):
    ''' Rescales an array to be between [0, 1]
    Args:
        arr_to_rescale:
    Returns:
        An array in [0,1] with f(0) = 0.5
    '''
    return ( arr_to_rescale / np.abs( arr_to_rescale ).max() ) / 2 + 0.5

def read_exr( image_fpath ):
    """ Reads an openEXR file into an RGB matrix with floats """
    f = OpenEXR.InputFile( image_fpath )
    dw = f.header()['dataWindow']
    w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
    im = np.empty( (h, w, 3) )

    # Read in the EXR
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = f.channels( ["R", "G", "B"], FLOAT )
    for i, channel in enumerate( channels ):
        im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
    return im


S3DIS_name_dict= {'chair':0, 'ceiling':1, 'column':2, 'table':3, 'window':4,  'sofa':5, 'wall':6, 'floor':7, 'board':8, 'door':9, 'bookcase':10, 'clutter':11, 'beam':12, '<UNK>':255}
json_label = load_labels('S3DIS_scipt/assets/semantic_labels.json')

def decode_gt_S3DIS(label_path, json_set = json_label):
    """
    :param label_path: path of a single label file
    :return: a decoded label map, should be ID matrix Uint8
    """

    label_map =  np.array(Image.open(label_path), dtype=np.uint32)
    h,w = label_map.shape[0:2]
    ret_map = np.zeros([h,w], dtype=np.uint8)
    Index_map = label_map[:,:,0] * 256 * 256 + label_map[:,:, 1] * 256 + label_map[:,:,2]
    for i in range(Index_map.shape[0]):
        for j in range(Index_map.shape[1]):
            index = Index_map[i][j]
            if int(index) > len(json_label):
                label_name = '<UNK>'
            else:
                label_name = parse_label(json_set[int(index)])['instance_class']
            label_ID = S3DIS_name_dict[label_name]
            ret_map[i][j] = label_ID
    return ret_map
#here to change