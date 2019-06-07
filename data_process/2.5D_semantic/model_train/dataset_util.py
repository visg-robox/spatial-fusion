#This is dataset_util for semantic segmentation task
#change this file to apply task on different task

import tensorflow as tf
from collections import namedtuple
import numpy as np

DATASET_SHOT = 'multi_sequence_road002'
DATA_DIR = '/media/luo/Dataset/RnnFusion/apollo_data/tfrecord/multi_sequence_road002'
NUM_IMAGES = {
    'train':1005,
    'validation':402,
}

CLASSNAME=[' sky ' ,
           ' car ' ,
           ' motorbicycle ' ,
           ' bicycle ' ,
           ' person ' ,
           ' rider ' ,
           ' truck ' ,
           ' bus ' ,
           ' tricycle ' ,
           ' road ' ,
           ' siderwalk ' ,
           ' traffic_cone ' ,
           ' road_pile ' ,
           ' fence ' ,
           ' traffic_light ' ,
           ' pole ' ,
           ' traffic_sign ' ,
           ' wall ' ,
           ' dustbin ' ,
           ' billboard ' ,
           ' building ' ,
           ' vegatation ']

NUM_CLASSES=len(CLASSNAME)

BLANCE_WEIGHT = np.array([1,1,1,1,1,1,1,1,1,0.2, 10, 10, 10, 3 , 10 , 5, 5,10,10,10, 1, 0.2],dtype = np.float)
BLANCE_WEIGHT = np.expand_dims(BLANCE_WEIGHT, axis = 1)


HEIGHT = 1344
WIDTH = 1664
IGNORE_LABEL = 255

RGB_MEAN = {'R' : 123.68, 'G' : 116.779, 'B' : 103.939}

LABEL_COLORS = [( 70, 130, 180),
                (  0,   0, 142),
                (  0,   0, 230),
                (119,  11,  32),
                (  0, 128, 192),
                (128,  64, 128),
                (128,   0, 192),
                (192,  0,  64),
                (128, 128, 192),
                (192, 128, 192),
                (192, 128,  64),
                (  0,   0,  64),
                (  0,   0, 192),
                ( 64,  64, 128),
                (192,  64, 128),
                (192, 128, 128),
                (  0,  64,  64),
                (192, 192, 128),
                ( 64,   0, 192),
                (192,   0, 192),
                (192,   0, 128),
                (128, 128,  64)]

#label colors calculate
# LABEL_COLORS=np.zeros([NUM_CLASSES,3],dtype=np.uint8)
# for label in labels:
#     if label.trainId<=22:
#     # color = (int(label.color[2:4],16),int(label.color[4:6],16),int(label.color[6:8],16))
#         color = label.color
#         r = color // (256 * 256)
#         g = (color - 256 * 256 * r) // 256
#         b = (color - 256 * 256 * r - 256 * g)
#         LABEL_COLORS[label.trainId]=np.array([r,g,b],dtype=np.uint8)



def parse_record(raw_record):
    """Parse apollo image and label from a tfrecord."""
    keys_to_features = {
        'label_raw':  tf.FixedLenFeature([], tf.string),
        'image_raw':  tf.FixedLenFeature([], tf.string),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    label = tf.decode_raw(parsed['label_raw'], tf.uint8)

    image = tf.cast(image, tf.float32)
    label=tf.cast(label,tf.int32)

    image.set_shape([HEIGHT *WIDTH* 3])
    image = tf.reshape(image, [HEIGHT ,WIDTH, 3])
    #bgr tfrecord
    bgr = tf.split(image, num_or_size_splits=3, axis=-1)
    image = tf.concat([bgr[2], bgr[1], bgr[0]], axis=-1)

    label.set_shape([HEIGHT * WIDTH * 1])
    label = tf.reshape(label, [HEIGHT,WIDTH, 1])

    return image, label



