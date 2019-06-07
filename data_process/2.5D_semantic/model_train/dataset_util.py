#This is dataset_util for semantic segmentation task
#change this file to apply task on different task

import tensorflow as tf
from collections import namedtuple

DATASET_SHOT = 'apollo_img_label_8000'
DATA_DIR = '../data_and_checkpoint/apollo_img_label_8000/tfrecord'
NUM_IMAGES = {
    'train':8000 ,
    'validation':1500,
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


Label = namedtuple('Label', [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class
    'clsId'       ,

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ])


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #     name                    clsId    id   trainId   category  catId  hasInstanceignoreInEval   color
    Label('others'              ,    0 ,    0,   255   , '其他'    ,   0  ,False , True  , 0x000000 ),
    Label('rover'               , 0x01 ,    1,   255   , '其他'    ,   0  ,False , True  , 0X000000 ),
    Label('sky'                 , 0x11 ,   17,    0    , '天空'    ,   1  ,False , False , 0x4682B4 ),
    Label('car'                 , 0x21 ,   33,    1    , '移动物体',   2  ,True  , False , 0x00008E ),
    Label('car_groups'          , 0xA1 ,  161,    1    , '移动物体',   2  ,True  , False , 0x00008E ),
    Label('motorbicycle'        , 0x22 ,   34,    2    , '移动物体',   2  ,True  , False , 0x0000E6 ),
    Label('motorbicycle_group'  , 0xA2 ,  162,    2    , '移动物体',   2  ,True  , False , 0x0000E6 ),
    Label('bicycle'             , 0x23 ,   35,    3    , '移动物体',   2  ,True  , False , 0x770B20 ),
    Label('bicycle_group'       , 0xA3 ,  163,    3    , '移动物体',   2  ,True  , False , 0x770B20 ),
    Label('person'              , 0x24 ,   36,    4    , '移动物体',   2  ,True  , False , 0x0080c0 ),
    Label('person_group'        , 0xA4 ,  164,    4    , '移动物体',   2  ,True  , False , 0x0080c0 ),
    Label('rider'               , 0x25 ,   37,    5    , '移动物体',   2  ,True  , False , 0x804080 ),
    Label('rider_group'         , 0xA5 ,  165,    5    , '移动物体',   2  ,True  , False , 0x804080 ),
    Label('truck'               , 0x26 ,   38,    6    , '移动物体',   2  ,True  , False , 0x8000c0 ),
    Label('truck_group'         , 0xA6 ,  166,    6    , '移动物体',   2  ,True  , False , 0x8000c0 ),
    Label('bus'                 , 0x27 ,   39,    7    , '移动物体',   2  ,True  , False , 0xc00040 ),
    Label('bus_group'           , 0xA7 ,  167,    7    , '移动物体',   2  ,True  , False , 0xc00040 ),
    Label('tricycle'            , 0x28 ,   40,    8    , '移动物体',   2  ,True  , False , 0x8080c0 ),
    Label('tricycle_group'      , 0xA8 ,  168,    8    , '移动物体',   2  ,True  , False , 0x8080c0 ),
    Label('road'                , 0x31 ,   49,    9    , '平面'    ,   3  ,False , False , 0xc080c0 ),
    Label('siderwalk'           , 0x32 ,   50,    10   , '平面'    ,   3  ,False , False , 0xc08040 ),
    Label('traffic_cone'        , 0x41 ,   65,    11   , '路间障碍',   4  ,False , False , 0x000040 ),
    Label('road_pile'           , 0x42 ,   66,    12   , '路间障碍',   4  ,False , False , 0x0000c0 ),
    Label('fence'               , 0x43 ,   67,    13   , '路间障碍',   4  ,False , False , 0x404080 ),
    Label('traffic_light'       , 0x51 ,   81,    14   , '路边物体',   5  ,False , False , 0xc04080 ),
    Label('pole'                , 0x52 ,   82,    15   , '路边物体',   5  ,False , False , 0xc08080 ),
    Label('traffic_sign'        , 0x53 ,   83,    16   , '路边物体',   5  ,False , False , 0x004040 ),
    Label('wall'                , 0x54 ,   84,    17   , '路边物体',   5  ,False , False , 0xc0c080 ),
    Label('dustbin'             , 0x55 ,   85,    18   , '路边物体',   5  ,False , False , 0x4000c0 ),
    Label('billboard'           , 0x56 ,   86,    19   , '路边物体',   5  ,False , False , 0xc000c0 ),
    Label('building'            , 0x61 ,   97,    20   , '建筑'    ,   6  ,False , False , 0xc00080 ),
    Label('bridge'              , 0x62 ,   98,    255  , '建筑'    ,   6  ,False , True  , 0x808000 ),
    Label('tunnel'              , 0x63 ,   99,    255  , '建筑'    ,   6  ,False , True  , 0x800000 ),
    Label('overpass'            , 0x64 ,  100,    255  , '建筑'    ,   6  ,False , True  , 0x408040 ),
    Label('vegatation'          , 0x71 ,  113,    21   , '自然'    ,   7  ,False , False , 0x808040 ),
    Label('unlabeled'           , 0xFF ,  255,    255  , '未标注'  ,   8  ,False , True  , 0xFFFFFF ),
]
