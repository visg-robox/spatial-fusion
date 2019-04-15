
import torch
from data_process.data_process import *
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
    if USING_RNN_FEATURE is True:
        res = torch.ones(batch_size, time_step, input_size) * 0.5
    if USING_SSNet_FEATURE is True:
        res = torch.zeros(batch_size, time_step, input_size + 1)
    for i in range(len(keys_list)):
        key = keys_list[i]
        related_feature = get_related_feature(key, voxel_map)
        feature_info = voxel_map[key].feature_info_list
        feature_len = len(feature_info)
        if USING_RNN_FEATURE:
            start_num = time_step - feature_len
            if start_num < 0:
                start_num = 0
            for j in range(start_num, time_step):
                feature_list = feature_info[j-start_num].feature_list
                # feature_list = None
                # count = 0
                # for k in range(len(related_feature)):
                #     if j < len(related_feature[k]):
                #         feature_list += related_feature[k].feature_list
                #         count += 1
                # feature_list = feature_list/count
                res[i][j] = torch.FloatTensor(feature_list)
        if USING_SSNet_FEATURE:
            start_num = 0
            end_num = feature_len + start_num
            if end_num > time_step:
                end_num = time_step
            for j in range(start_num, end_num):
                # feature_list = numpy.append(1, feature_info[j - start_num].feature_list)
                feature_list = numpy.zeros_like(numpy.append(1, feature_info[j - start_num].feature_list))
                count = 0
                for k in range(len(related_feature)):
                    if j < len(related_feature[k]):
                        feature_list = feature_list + numpy.append(1, related_feature[k][j].feature_list)
                        count += 1
                feature_list = feature_list/count
                res[i][j] = torch.FloatTensor(feature_list)
    return res


# get batch tensor in number form
def featuremap_to_gt_num(voxel_map, keys_list, batch_size):
    res = torch.zeros(batch_size, dtype=torch.int64)
    for i in range(len(keys_list)):
        key = keys_list[i]
        semantic_info = voxel_map[key].feature_info_list
        # no effect?
        res[i] = int(semantic_info[0].feature_list[0])
        # res[i] = torch.FloatTensor(semantic_info[0].label_list)
    return res


def get_related_feature(key, voxel_map):
    center = key_to_center(key)
    related_feature = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                current_center = np.array([center[0]+x*common.voxel_length,
                                           center[1]+y*common.voxel_length,
                                           center[2]+z*common.voxel_length])
                current_key = center_to_key(current_center)
                # current_feature = voxel_map.find(current_key).feature_info_list
                if current_key in voxel_map.keys():
                    current_feature = voxel_map[current_key].feature_info_list
                    related_feature.append(current_feature)
    return related_feature


if __name__ == '__main__':
    pass





