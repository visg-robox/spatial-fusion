import sys
sys.path.append('/data1/zhangjian/spatial-fusion/')
import os, shutil
import common
import numpy as np


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

if __name__ == "__main__":
    data_path = common.blockfile_path
    infer_path = data_path + 'infer_feature/'
    infer_label_path = data_path + 'infer_label/'
    gt_path = data_path + 'gt_feature/'
    test_infer_path = data_path + 'test_feature/infer/'
    test_gt_path = data_path + 'test_feature/gt/'
    test_label_path = data_path + 'test_feature/infer_label/'
    make_path(test_infer_path)
    make_path(test_gt_path)
    make_path(test_label_path)

    #initial_pose = np.loadtxt('./raw.txt')
    initial_pose = np.array([-1.206e+04, 2.755e+03, 3.997e+01])
    infer_file = get_file_list(infer_path)
    infer_file.sort(key=lambda x:(to_distance(x.split('/')[-1].split('.')[0], initial_pose)))
    infer_label_file = get_file_list(infer_label_path)
    infer_label_file.sort(key=lambda x: (to_distance(x.split('/')[-1].split('.')[0], initial_pose)))
    gt_file = get_file_list(gt_path)
    gt_file.sort(key=lambda x:(to_distance(x.split('/')[-1].split('.')[0], initial_pose)))

    divide_num = 5
    each_divide = len(infer_file) // divide_num
    move_length = each_divide // 5

    for i in range(divide_num):
        infer_file_move = infer_file[(i + 1) * each_divide - move_length: (i + 1) * each_divide]
        gt_file_move = gt_file[(i + 1) * each_divide - move_length: (i + 1) * each_divide]
        infer_label_move = infer_label_file[(i + 1) * each_divide - move_length: (i + 1) * each_divide]
        for item in infer_file_move:
            cur_infer_file = item
            fpath, fname = os.path.split(cur_infer_file)
            shutil.move(cur_infer_file, test_infer_path + fname)

        for item in gt_file_move:
            cur_gt_file = item
            fpath, fname = os.path.split(cur_gt_file)
            shutil.move(cur_gt_file, test_gt_path + fname)

        for item in infer_label_move:
            cur_label_file = item
            fpath, fname = os.path.split(cur_label_file)
            shutil.move(cur_label_file, test_label_path + fname)

