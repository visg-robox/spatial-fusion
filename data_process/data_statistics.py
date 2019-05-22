import numpy as np
import os
from os.path import join


CLASS_NUM = 13
SAMPLE_NUM = 20000

label_cal = np.zeros([13], dtype = np.float64())

abspath = '/media/luo/Dataset/RnnFusion/test3/gt_feature'
path_list = os.listdir(abspath)

def get_probabilty(ratio, num, sample_num):
    max_ratio = np.max(ratio)
    max_num = np.max(num)
    sample_ratio = sample_num / max_num
    probabilty = sample_ratio * (max_ratio / ratio)
    return probabilty


if __name__ == '__main__':
    num = 0
    for path in path_list:
        num += 1
        if(num % 5 == 0):
            print(num)
        a = np.load(join(abspath, path))
        voxel_list = a.item().values()
        for i in voxel_list:
            label = i.feature_info_list[0].feature_list
            label_cal[int(label)] += 1

    total_num = np.sum(label_cal)
    print(total_num)
    ratio = label_cal / total_num
    probabilty = get_probabilty(ratio, num, SAMPLE_NUM)
    ratio = np.concatenate([np.expand_dims(ratio,axis=0), np.expand_dims(label_cal,axis=0)], axis = 0)
    np.savetxt('data_ratio.txt',ratio,fmt='%.3e',delimiter='\t')
    np.savetxt('data_dropout_ratio.txt',probabilty,fmt='%.3e',delimiter='\t')

