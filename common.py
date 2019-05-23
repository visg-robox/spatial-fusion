from enum import Enum, unique
import itertools
import math

voxel_length = 0.05
class_num = 22
ignore_list = [0, 4, 12]

# input feature order
img_feature_size = 128
vector_size = 3
offset_size = 3
location_size = 3


feature_num_raw = img_feature_size
feature_num_new = img_feature_size + vector_size
feature_num = img_feature_size + vector_size + offset_size
qk_dim = 256



# region
region_x = 10
region_y = 20
region_z = 2
block_len = 5

#
batch_size = 512
time_step = 20



# fusion method
@unique
class FsMethod(Enum):
    RNN_FEATURE = 0
    RNN_LABEL = 1
    BAYES = 2
    BASELINE = 3
    GT = 4
    STF = 5

# tag
USING_RNN_FEATURE = False
USING_SSNet_FEATURE = True
USING_RNN = False
USING_SSNet = True


# path definition
project_path = "/home/wangkai/spatial-fusion/"
data_path = "/home/wangkai/project2/RnnFusion/data/"

# related space

offset_list = []
offset = 2
#near_num = int(math.pow((offset*2+1), 3))
near_num = 25
for i in itertools.product([i-2 for i in range(2 * offset + 1)], repeat=3):
    offset_list.append(i)
