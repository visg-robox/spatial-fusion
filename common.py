import os
from enum import Enum, unique
import sys


def make_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def load_txt_dict(txt_path):
    with open(txt_path) as f:
        para_dict = dict()
        line = f.readline()
        while line:
            if line is not '':
                line_list = line.split(':')
                para_dict[line_list[0]] = line_list[-1][:-1]
            line = f.readline()
    return para_dict


#txt_path = 'record002.txt'
txt_path = sys.argv[1]
para_dict = load_txt_dict(txt_path)




# preprocess config path ####################################################
voxel_length = 0.05
# region
block_len = 5
region_x = int(para_dict['region_x'])
region_y = int(para_dict['region_y'])
region_z = int(para_dict['region_z'])

lidardata_path = para_dict['lidardata_path']
blockfile_path = para_dict['blockfile_path']
class_preserve_proba_path = os.path.join(blockfile_path, para_dict['class_preserve_proba_path'])


# ###########################################################################


# dataset config path #######################################################

dataset_name = para_dict['dataset_name']
class_num = int(para_dict['class_num'])
ignore_list_str = para_dict['ignore_list_str'].split()
ignore_list = [int(ignore_list_str[i]) for i in range(len(ignore_list_str))]


# ############################################################################


# train&test config path #####################################################

near_num = int(para_dict['near_num'])
# input feature order
img_feature_size = 128
vector_size = 3
offset_size = 3
location_size = 3
# feature size
feature_num_i = img_feature_size
feature_num_iv = img_feature_size + vector_size
feature_num_ivo = img_feature_size + vector_size + offset_size
qk_dim = 256

# train parameter
epoch = int(para_dict['epoch'])
lr = float(para_dict['lr'])
batch_size = int(para_dict['batch_size'])
time_step = int(para_dict['time_step'])
pretrained = bool(int(para_dict['pretrained']))
model_path = para_dict['model_path']
model_dir = para_dict['model_dir']

file_num_step = int(para_dict['file_num_step'])

# test parameter
test_batch_size = int(para_dict['test_batch_size'])
test_model_path = para_dict['test_model_path']

# save
model_save_step = int(para_dict['model_save_step'])
res_save_path = 'result'

# ############################################################################


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





