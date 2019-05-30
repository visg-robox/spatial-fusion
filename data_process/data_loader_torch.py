
import torch
import math
from scipy import stats
from data_process.data_process_label import *
from common import USING_RNN_FEATURE, USING_SSNet_FEATURE, USING_RNN, USING_SSNet


# input:
#   voxel_map: data source
# output:
#   batch tensor, shape: (batch_size, time_step, input_size + 1)
def labelmap_to_batch(voxel_map, keys_list, batch_size, time_step, input_size):

    if USING_SSNet is True:
        res = torch.zeros(batch_size, time_step, input_size + 1)
    if USING_RNN is True:
        res = torch.ones((batch_size, time_step, input_size)) * 0.5
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = voxel_map[key].semantic_info_list
        label_len = len(semantic_info)
        if USING_SSNet:
            start_num = 0
            end_num = label_len + start_num
            if end_num > time_step:
                end_num = time_step
            for j in range(start_num, end_num):
                label_list = numpy.append(1, semantic_info[j - start_num].label_list)
                res[i][j] = torch.FloatTensor(label_list)
        if USING_RNN:
            start_num = time_step - label_len
            if start_num < 0:
                start_num = 0
            for j in range(start_num, time_step):
                label_list = semantic_info[j - start_num].label_list
                res[i][j] = torch.FloatTensor(label_list)
            # end_num = label_len
            # if end_num > time_step:
            #     end_num = time_step
            # for j in range(end_num):
            #     label_list = semantic_info[j].label_list
            #     res[i][j] = torch.FloatTensor(label_list)
    return res


#  get batch tensor as one hot form
def labelmap_to_gt_onehot(voxel_map, keys_list, batch_size, input_size):
    res = torch.zeros(batch_size, input_size, dtype=torch.float32)
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = voxel_map[key].semantic_info_list
        label_len = len(semantic_info)
        for j in range(label_len):
            class_num = semantic_info[j].label_list
            one_hot = torch.zeros(input_size)
            one_hot[class_num] = 1
            res[i] += one_hot
        res[i] = res[i] / label_len
    return res


# get batch tensor in number form
def labelmap_to_gt_num(voxel_map, keys_list, batch_size):
    res = torch.zeros(batch_size, dtype=torch.int64)
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = voxel_map[key].semantic_info_list
        # no effect?
        res[i] = semantic_info[0].label_list[0]
        # res[i] = torch.FloatTensor(semantic_info[0].label_list)
    return res


# ########################## feature map ##########################

#
def featuremap_to_batch(voxel_map, keys_list, batch_size, time_step, input_size):
    res = torch.zeros(batch_size, time_step, input_size + 1)
    for i in range(len(keys_list)):
        key = keys_list[i]
        if len(key) != 3:
            key = tuple(list(key)[:3])
        feature_info = voxel_map[key].feature_info_list
        feature_len = len(feature_info)
        start_num = 0
        end_num = feature_len + start_num
        if end_num > time_step:
            end_num = time_step
        for j in range(start_num, end_num):
            # feature_list = numpy.append(1, feature_info[j - start_num].feature_list)
            feature_list = numpy.zeros_like(numpy.append(1, feature_info[j - start_num].feature_list))
            res[i][j] = torch.FloatTensor(feature_list)
    return res


def featuremap_to_batch_v(voxel_map, keys_list, batch_size, time_step, input_size):
    res = torch.zeros(batch_size, time_step, input_size + 1)
    for i in range(len(keys_list)):
        key = keys_list[i]
        feature_info = voxel_map[key].feature_info_list
        feature_len = len(feature_info)
        start_num = 0
        end_num = feature_len + start_num
        if end_num > time_step:
            end_num = time_step
        for j in range(start_num, end_num):
            # feature_list = numpy.append(1, feature_info[j - start_num].feature_list)
            feature_list = numpy.zeros_like(numpy.append(1, feature_info[j - start_num].feature_list, feature_info[j - start_num].vector))
            res[i][j] = torch.FloatTensor(feature_list)
    return res


def featuremap_to_batch_with_distance(voxel_map, keys_list, batch_size, near_num, time_step, input_size):
    res = torch.zeros(batch_size, near_num, time_step, input_size + 1)
    for i in range(len(keys_list)):                                                  # batch dim
        key = keys_list[i]
        #related_keys
        related_feature = get_related_voxels(key, voxel_map)
        for j in range(len(related_feature)):# near_num dim
            offset_vector = index_to_offset(j, offset)
            feature_info = related_feature[j]
            if(feature_info != None):
                feature_len = len(feature_info)
                start_num = 0
                end_num = feature_len + start_num
                if end_num > time_step:
                    end_num = time_step

                for k in range(start_num, end_num):                                  # time_step dim
                    feature_list = [1] + list(feature_info[k - start_num].feature_list) + \
                                   feature_info[k - start_num].vector + offset_vector  # list(feature_info[k - start_num].vector
                    res[i][j][k] = torch.FloatTensor(feature_list)
    return res


def featuremap_to_batch_with_balance(voxel_map, keys_list, batch_size, near_num, time_step, input_size):
    res = torch.zeros(batch_size, near_num, time_step, input_size + 1)
    for i in range(len(keys_list)):                                                  # batch dim
        # len(keys_list == batch_size)
        key = keys_list[i]
        #related_keys
        if len(key) != 3:
            key = tuple(list(key)[:3])
        related_feature = voxel_map[key]
        for j in range(len(related_feature)):                                        # near_num dim
            related_key = center_to_key(related_feature[j].center)
            offset_vector = [key[m] - related_key[m] for m in range(3)]
            feature_info = related_feature[j].feature_info_list
            if(feature_info != None):
                feature_len = len(feature_info)
                start_num = 0
                end_num = feature_len + start_num
                if end_num > time_step:
                    end_num = time_step

                for k in range(start_num, end_num):                                  # time_step dim
                    feature_list = [1] + list(feature_info[k - start_num].feature_list) + \
                                   feature_info[k - start_num].vector + offset_vector  # list(feature_info[k - start_num].vector
                    res[i][j][k] = torch.FloatTensor(feature_list)
    return res


def index_to_offset(index, offset):
    offset_vector = []
    for i in range(3):
        tmp = int(math.pow(2*offset+1, 2-i))
        num = index // tmp
        offset_vector.append(num-offset)
        index -= num*tmp
    return offset_vector


# get batch tensor in number form
def featuremap_to_gt_num(voxel_map, keys_list, batch_size, ignore_list):
    res = torch.zeros(batch_size, dtype=torch.int64)
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = voxel_map[key].feature_info_list
        # no effect?
        gt_list = list()
        for j in range(len(semantic_info)):
            gt_list.append(int(semantic_info[j].feature_list[0]))
        res[i] = int(stats.mode(gt_list)[0][0])
        # res[i] = int(semantic_info[0].feature_list[0])
        #这里把无效的class置为255
        if res[i] in ignore_list:
            res[i] = int(-100)
        # res[i] = torch.FloatTensor(semantic_info[0].label_list)

    return res


def get_related_voxels(key, voxel_map):
    related_feature = []
    center = key_to_center(key)
    for item in common.offset_list:
        related_center = [center[i] + list(item)[i]*common.voxel_length for i in range(len(key))]   # why need center_to_key?
        related_key = center_to_key(related_center)
        if related_key in voxel_map:
            related_feature.append(voxel_map[related_key].feature_info_list)
        else:
            related_feature.append(None)

    return related_feature


if __name__ == '__main__':
    pass





