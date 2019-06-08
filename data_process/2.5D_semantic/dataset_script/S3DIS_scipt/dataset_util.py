#This is dataset_util for semantic segmentation task
#change this file to apply task on different task

import tensorflow as tf
import tensorflow as tf
from collections import namedtuple
import numpy as np

DATASET_SHOT = 'S3DIS_Sample0.2'
DATA_DIR = '/media/luo/Dataset/S3DIS/S3DIS_tfrecord'
NUM_IMAGES = {
    'train':10580 ,
    'validation':3580,
}

CLASSNAME=['chair', 'ceiling', 'column', 'table', 'window',  'sofa', 'wall', 'floor', 'board', 'door', 'bookcase', 'clutter', 'beam']
NUM_CLASSES=len(CLASSNAME) #13

HEIGHT = 1080
WIDTH = 1080
IGNORE_LABEL = 255


RGB_MEAN = {'R' : 123.68, 'G' : 116.779, 'B' : 103.939}

# colour map
LABEL_COLORS = [
    [255, 255, 255],
    [220, 20, 60],
    [190, 153, 153],
    [0, 0, 0],
    [70, 70, 70],
    [0, 255, 255],
    [255, 255, 0],
    [0, 0, 255],
    [244, 35, 232],
    [107, 142, 35],
     [151, 115, 255],
     [102, 102, 156],
     [255, 124, 0]]

BLANCE_WEIGHT = np.ones([NUM_CLASSES], dtype = np.float32)
BLANCE_WEIGHT = np.expand_dims(BLANCE_WEIGHT, axis = 1)

def parse_record(raw_record):
    """Parse kitti image and label from a tf record."""
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

    label.set_shape([HEIGHT * WIDTH * 3])
    label = tf.reshape(label, [HEIGHT,WIDTH, 3])
    label = tf.slice(label,(0,0,0),(HEIGHT,WIDTH,1))
    print(label)

    return image, label
