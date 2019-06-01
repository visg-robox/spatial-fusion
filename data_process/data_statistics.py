import numpy as np
import os
from os.path import join
import common

CLASS_NUM = common.class_num
SAMPLE_ratio = 0.2





def get_probabilty(ratio, num, sample_ratio, min_num = 100):

    probabilty = np.zeros_like(ratio, dtype= np.float)

    for i in range(len(list(num))):
        if num[i] < min_num:
            probabilty[i] = 0
        else:
            probabilty[i] = sample_ratio / (np.sum(np.greater(num, min_num) * ratio[i]))

    return probabilty

def statistics(data_path, classnum, save_path):
    label_cal = np.zeros([classnum], dtype=np.float64())
    path_list = os.listdir(data_path)
    num = 0
    for path in path_list:
        num += 1
        if (num % 5 == 0):
            print(num)
        a = np.load(join(data_path, path))
        voxel_list = a.item().values()
        for i in voxel_list:
            label = i.feature_info_list[0].feature_list
            label_cal[int(label)] += 1

    total_num = np.sum(label_cal)
    print(total_num)
    ratio = label_cal / total_num
    ratio = np.concatenate([np.expand_dims(ratio, axis=0), np.expand_dims(label_cal, axis=0)], axis=0)
    np.savetxt(save_path, ratio, fmt='%.3e', delimiter='\t')

if __name__ == '__main__':
    STATISTICS = False
    PROBABILTY = True
    if(STATISTICS):
        gt_path = os.path.join(common.blockfile_path, 'gt')
        save_path = os.path.join(common.blockfile_path, 'data_ratio.txt')
        statistics(gt_path, CLASS_NUM, save_path)


    if(PROBABILTY):
        ratio_path = os.path.join(common.blockfile_path, 'data_ratio.txt')
        statistic = np.loadtxt(ratio_path)
        ratio = statistic[0, :]
        num = statistic[1, :]
        probabilty = get_probabilty(ratio, num, SAMPLE_ratio)
        save_path2 = os.path.join(common.blockfile_path, 'data_dropout_ratio.txt')
        np.savetxt(save_path2, probabilty, fmt='%.3e', delimiter='\t')

