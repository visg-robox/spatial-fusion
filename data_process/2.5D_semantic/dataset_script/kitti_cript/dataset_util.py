#This is dataset_util for semantic segmentation task
#change this file to apply task on different task

import tensorflow as tf


DATA_DIR = '../data_and_checkpoint/kitti/tfrecord'
NUM_IMAGES = {
    'train':157 ,
    'validation':43,
}

CLASSNAME=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motocycle','bicycle']
NUM_CLASSES=len(CLASSNAME)

HEIGHT = 375
WIDTH = 1242
IGNORE_LABEL = 255


RGB_MEAN = {'R' : 123.68, 'G' : 116.779, 'B' : 103.939}

# colour map
LABEL_COLORS = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32)]
                # 18 = bicycle, 19 = void label

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

    label.set_shape([HEIGHT * WIDTH * 1])
    label = tf.reshape(label, [HEIGHT,WIDTH, 1])

    return image, label
