
import os
import time
import tensorflow as tf
import numpy
import common


# return 3d numpy tensor
def labelmap_to_batch(map, keys_list, batch_size, time_step, input_size):
    res = numpy.ones((batch_size, time_step, input_size)) * 0.5
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = map.find_key(key).semantic_info_list
        label_len = len(semantic_info)
        start_num = time_step - label_len
        if start_num < 0:
            start_num = 0
        for j in range(start_num, time_step):
            res[i][j] = semantic_info[j-start_num].label_list
    return res


def labelmap_to_gt_onehot(map, keys_list, batch_size, input_size):
    res = numpy.zeros((batch_size, input_size))
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = map.find_key(key).semantic_info_list
        label_len = len(semantic_info)
        for j in range(label_len):
            class_num = semantic_info[j].label_list
            one_hot = numpy.zeros(input_size)
            one_hot[int(class_num)] = 1
            res[i] += one_hot
        res[i] = res[i] / label_len
    return res


def labelmap_to_gt_num(map, keys_list, batch_size):
    res = numpy.zeros(batch_size)
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = map.find_key(key).semantic_info_list
        res[i] = semantic_info[0].label_list[0]
    return res


#
def hashmap_to_onehot(hashmap, input_size):
    keys_list = list(hashmap.keys())
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = hashmap.find_key(key).semantic_info_list
        label_len = len(semantic_info)
        for j in range(label_len):
            class_num = int(semantic_info[j].label_list)
            one_hot = numpy.zeros(input_size)
            one_hot[int(class_num)] = 1
            semantic_info[j].label_list = one_hot


if __name__ == '__main__':
    pass
    # data_path = '/home/zhangjian/code/data/CARLA_episode_0019/test/'
    # pointPath = data_path + 'infer/'
    # hashmap = pointlist_to_hashmap(pointPath)
    # print('here')
    # keys_list = list(hashmap.keys())
    # keys_list = keys_list[0:100]
    # labelmap_to_batch(hashmap, keys_list, 100, 50, 13)

    # data_path = '/home/zhangjian/code/data/CARLA_episode_0019/test1/'
    # gt_path = data_path + 'test/gt/'
    # gt_hashmap = pointlist_to_hashmap(gt_path)
    # hashmap_to_onehot(gt_hashmap, 13)
    # print('1')




