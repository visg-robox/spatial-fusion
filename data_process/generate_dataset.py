"""
generate label/feature

"""
import sys
sys.path.append("../")
import os, shutil
import common
import numpy as np
from data_process.data_statistics import save_preserve_ratio
from data_process.data_process_feature import preprocess_feature, preprocess_record_feature

SCENCE_NUM = 4


def get_file_list(data_dir):
    path_list = list()
    for i in os.listdir(data_dir):
        path_list.append(os.path.join(data_dir, i))
    return path_list


def make_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def to_distance(name, initial_pose):
    tmp = np.zeros(initial_pose.shape)
    for i in range(len(name.split('_'))):
        tmp[i] = int(name.split('_')[i])
    return np.linalg.norm(tmp - initial_pose)


def devide_with_pose():
    initial_pose = preprocess_feature()
    initial_pose = np.array([-120, -120, 40])
    lidardata_path = common.blockfile_path
    infer_feature_path = os.path.join(lidardata_path, 'infer_feature')
    infer_label_path = os.path.join(lidardata_path, 'infer_label')
    gt_path = os.path.join(lidardata_path, 'gt')

    test_infer_save_path = os.path.join(common.blockfile_path, 'test', 'infer_feature')
    test_p_save_path = os.path.join(common.blockfile_path, 'test', 'infer_label')
    test_gt_save_path = os.path.join(common.blockfile_path, 'test', 'gt')

    make_path(test_infer_save_path)
    make_path(test_gt_save_path)
    make_path(test_p_save_path)

    # initial_pose = np.zeros([3])
    infer_feature_file = get_file_list(infer_feature_path)
    infer_feature_file.sort(key=lambda x:(to_distance(x.split('/')[-1].split('.')[0], initial_pose)))
    infer_label_file = get_file_list(infer_label_path)
    infer_label_file.sort(key=lambda x:(to_distance(x.split('/')[-1].split('.')[0], initial_pose)))
    gt_file = get_file_list(gt_path)
    gt_file.sort(key=lambda x:(to_distance(x.split('/')[-1].split('.')[0], initial_pose)))

    scene = len(infer_feature_file)//SCENCE_NUM
    scene_len = len(infer_feature_file)//(SCENCE_NUM * 5)
    infer_feature_file_move = []
    infer_label_file_move = []
    gt_file_move = []
    for i in range(SCENCE_NUM):
        infer_feature_file_move = infer_feature_file_move + infer_feature_file[scene*(i+1)-scene_len:scene*(i+1)]
        infer_label_file_move = infer_label_file_move + infer_label_file[scene*(i+1)-scene_len:scene*(i+1)]
        gt_file_move = gt_file_move + gt_file[scene*(i+1)-scene_len:scene*(i+1)]

    for item in infer_feature_file_move:
        cur_infer_feature_file = item
        fpath, fname = os.path.split(cur_infer_feature_file)
        shutil.move(cur_infer_feature_file, os.path.join(test_infer_save_path, fname))

    for item in infer_label_file_move:
        cur_infer_label_file = item
        fpath, fname = os.path.split(cur_infer_label_file)
        shutil.move(cur_infer_label_file, os.path.join(test_p_save_path, fname))

    for item in gt_file_move:
        cur_gt_file = item
        fpath, fname = os.path.split(cur_gt_file)
        shutil.move(cur_gt_file, os.path.join(test_gt_save_path, fname))


def divide_multi_sequence():
    lidardata_path = common.lidardata_path
    blockfile_path = common.blockfile_path
    train_list = common.train_sequence_list
    test_list = common.test_sequence_list
    for item in train_list:
        cur_lidardata_path = os.path.join(lidardata_path, item)
        cur_save_path = os.path.join(blockfile_path, 'train', item)
        common.make_path(cur_save_path)
        preprocess_record_feature(cur_lidardata_path, cur_save_path)
    for item in test_list:
        cur_lidardata_path = os.path.join(lidardata_path, item)
        cur_save_path = os.path.join(blockfile_path, 'test', item)
        common.make_path(cur_save_path)
        preprocess_record_feature(cur_lidardata_path, cur_save_path)
    save_preserve_ratio()


def divide_multi_sequence_s3d():
    lidardata_path = common.lidardata_path
    blockfile_path = common.blockfile_path
    train_list = os.listdir(os.path.join(common.blockfile_path, 'train'))
    test_list = os.listdir(os.path.join(common.test_sequence_list, 'test'))
    for item in train_list:
        cur_lidardata_path = os.path.join(lidardata_path, item)
        cur_save_path = os.path.join(blockfile_path, 'train', item)
        common.make_path(cur_save_path)
        preprocess_record_feature(cur_lidardata_path, cur_save_path)
    for item in test_list:
        cur_lidardata_path = os.path.join(lidardata_path, item)
        cur_save_path = os.path.join(blockfile_path, 'test', item)
        common.make_path(cur_save_path)
        preprocess_record_feature(cur_lidardata_path, cur_save_path)
    save_preserve_ratio()


if __name__ == "__main__":
    divide_multi_sequence()


