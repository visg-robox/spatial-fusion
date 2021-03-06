from enum import Enum, unique

voxel_length = 0.05
class_num = 13
feature_num = 128
# region
region_x = 20
region_y = 20
region_z = 10
block_len = 5

#
batch_size = 512
time_step = 50


# fusion method
@unique
class FsMethod(Enum):
    RNN_FEATURE = 0
    RNN_LABEL = 1
    BAYES = 2
    BASELINE = 3


# tag
USING_RNN_FEATURE = False
USING_SSNet_FEATURE = True
USING_RNN = False
USING_SSNet = True


# path definition
project_path = "/home/zhangjian/code/project/spatial-fusion/"
data_path = "/home/zhangjian/code/project/data/"

