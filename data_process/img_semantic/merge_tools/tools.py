import scipy.io as sio
import numpy as np
from PIL import Image
import tensorflow as tf

label_colours = [[128, 128, 128], [128, 0, 0], [128, 64, 128]
                # 0 = sky, 1 = Building, 2 = road
                ,[0, 0, 192], [64, 64, 128], [128, 128, 0]
                # 3 = sidewalk, 4 = fence, 5 = vegetation
                ,[192, 192, 128], [64, 0, 128], [192, 128, 128]
                # 6 = pole, 7 = car, 8 = traffic sign
                ,[64, 64, 0], [0, 128, 192], [0, 172, 0],[0, 60, 100]]
                # 9 = pedestrain, 10 = bicycle, 11 = lanemarking

                # 12 = rider, 13 = car, 14 = truck
                #,[0, 60, 100],[0, 79, 100], [0, 0, 230]
                
                # 15 = bus, 16 = train, 17 = motocycle
                #,[119, 10, 32], [65, 0, 110],
                # 18 = bicycle, 19 = void label
                #[89,56,0],[120,189,10],[56,0,160],[150,10,56],[70,10,220],[90,200,261],[180,30,150],[46,120,120]]
def decode_labels(mask, img_shape, num_classes):
    color_mat = tf.constant(label_colours, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))
    
    return pred


def decode_pclabel(mask):
    color_mat = tf.constant(label_colours[1:12], dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=11)
    onehot_output = tf.reshape(onehot_output, (-1,11))
    pred = tf.matmul(onehot_output, color_mat)
    return(pred)

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch
