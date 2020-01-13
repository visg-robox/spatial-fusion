import sys
sys.path.append("../")

import numpy as np
import os
from os.path import join
import common

CLASS_NUM = common.class_num
SAMPLE_ratio = 0.2
MIN_CLASS_NUM = 10
from scipy import stats
import glob
import common


def get_probabilty(ratio, num, sample_ratio, min_num=MIN_CLASS_NUM):
    probabilty = np.zeros_like(ratio, dtype=np.float)
    
    for i in range(len(list(num))):
        if num[i] < min_num:
            probabilty[i] = 0
        else:
            probabilty[i] = sample_ratio / (np.sum(np.greater(num, min_num) * ratio[i]))
    
    return probabilty


def statistics(path_list, classnum, save_path):
    label_cal = np.zeros([classnum], dtype=np.float64())
    num = 0
    for path in path_list:
        num += 1
        if (num % 5 == 0):
            print(num)
        a = np.load(path)
        voxel_list = a.item().values()
        for i in voxel_list:
            semantic_info = i.feature_info_list
            gt_list = list()
            for j in range(len(semantic_info)):
                gt_list.append(int(semantic_info[j].feature_list[0]))
            gt_class = int(stats.mode(gt_list)[0][0])
            if gt_class < classnum and gt_class >= 0:
                label_cal[int(gt_class)] += 1
    total_num = np.sum(label_cal)
    print(total_num)
    ratio = label_cal / total_num
    ratio = np.concatenate([np.expand_dims(ratio, axis=0), np.expand_dims(label_cal, axis=0)], axis=0)
    np.savetxt(save_path, ratio, fmt='%.3e', delimiter='\t')


def save_preserve_ratio(sample_ratio=0.2, min_classnum=100):
    STATISTICS = True
    PROBABILTY = True
    if (STATISTICS):

        gt_path_list = glob.glob(join(common.blockfile_path, '*/*/gt/*.npy'))
        save_path = os.path.join(common.blockfile_path, 'data_ratio.txt')
        statistics(gt_path_list, CLASS_NUM, save_path)
    
    if (PROBABILTY):
        ratio_path = os.path.join(common.blockfile_path, 'data_ratio.txt')
        statistic = np.loadtxt(ratio_path)
        ratio = statistic[0, :]
        num = statistic[1, :]
        probabilty = get_probabilty(ratio, num, sample_ratio, min_classnum)
        save_path2 = os.path.join(common.blockfile_path, 'data_dropout_ratio.txt')
        np.savetxt(save_path2, probabilty, fmt='%.3e', delimiter='\t')


if __name__ == '__main__':
    save_preserve_ratio()
