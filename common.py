import os
from enum import Enum, unique
import sys
import glob
import numpy as np

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


def get_common_keys(infer_dict, gt_dict):
    infer_keys_list = list(infer_dict.keys())
    gt_keys_list = list(gt_dict.keys())
    # common_keys_list = [v for v in infer_keys_list if v in gt_keys_list]
    return infer_keys_list


def get_file_list(data_dir):
    path_list = list()
    for i in os.listdir(data_dir):
        path_list.append(os.path.join(data_dir, i))
    return path_list


def find_file_with_pattern(pattern, path='.'):
    matches = []
    dirs = []
    res = []
    for x in os.listdir(path):
        nd = os.path.join(path, x)
        if os.path.isdir(nd):
            dirs.append(nd)
        elif os.path.isfile(nd) and pattern in x:
            matches.append(nd)
    for match in matches:
        res.append(match)
    for dir in dirs:
        res = res + find_file_with_pattern(pattern, path=dir)
    return res


def find_dir_with_pattern(pattern, path='.'):
    matches = []
    dirs = []
    res = []
    for x in os.listdir(path):
        nd = os.path.join(path, x)
        if os.path.isdir(nd) and pattern in x:
            matches.append(nd)
        elif os.path.isdir(nd):
            dirs.append(nd)
    for match in matches:
        res.append(match)
    for dir in dirs:
        res = res + find_dir_with_pattern(pattern, path=dir)
    return res


def get_file_list_with_pattern(pattern, dir, sorted=True):
    dir_list = find_dir_with_pattern(pattern, dir)
    file_list = []
    for dir in dir_list:
        file_list = file_list + get_file_list(dir)
    if sorted:
        file_list.sort()
    return file_list



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


if int(para_dict['multi_sequence']):
    raw_data_path = para_dict['raw_data_path']
    train_sequence_list = para_dict['train_sequence_list'].split(' ')
    test_sequence_list = para_dict['test_sequence_list'].split(' ')
    point_num_per_frame = int(para_dict['point_num_per_frame'])
    frame_num_pre_sequence = int(para_dict['frame_num_pre_sequence'])


# ###########################################################################


# dataset config path #######################################################

dataset_class_config = para_dict['dataset_class_config']
dataset_name = para_dict['dataset_name']
class_num = int(para_dict['class_num'])

if os.path.exists(class_preserve_proba_path):
    data_preserve_ratio = np.loadtxt(class_preserve_proba_path)
    ignore_list = list(np.where(np.equal(data_preserve_ratio,0))[0])

# ignore_list_str = para_dict['ignore_list_str'].split()
# ignore_list = [int(ignore_list_str[i]) for i in range(len(ignore_list_str))]


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

file_num_step = int(para_dict['file_num_step'])

# test parameter
test_batch_size = int(para_dict['test_batch_size'])

# save
model_save_step = int(para_dict['model_save_step'])
res_save_path = para_dict['res_save_path']
make_path(res_save_path)
# ############################################################################
if sys.argv[2] == 'train':
    method_name = (sys.argv[0]).split('/')[-1].split('.')[0]  # pretrain
    pre_train_model_dir = os.path.join(res_save_path, dataset_name, method_name)
    if pretrained:
        pre_train_step = sys.argv[3]
        pre_train_model_path = os.path.join(pre_train_model_dir, pre_train_step + '_model.pkl')
if sys.argv[2] == 'test':
    method_name = sys.argv[3]  # pretrain & eval
    test_model_path = os.path.join(res_save_path, dataset_name, method_name)

# fusion method
@unique
class FsMeth.od(Enum):
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





