"""
input:voxel_map(block), gt_map(block), keys_map(related keys), lable_probability
output:dict(key, voxel(cur_voxel, related_voxel))
voxel_array: [near_num+1, ]
"""
# import numpy as np
from data_structure.voxel_map import *
from sklearn.neighbors import KDTree

NEAR_NUM = 25
MAX_DISTANCE = np.linalg.norm(center_to_key([1, 1, 1]), 2)


def data_balance_rnn(voxel_map, gt_map, label_probability):
    keys = common.get_common_keys(voxel_map, gt_map)
    keys_list = []
    for i in range(len(keys)):
        key = keys[i]
        label = int(gt_map[key].feature_info_list[0].feature_list[0])
        if label < common.class_num and label >= 0:
            probability = label_probability[label]
        else:
            probability = 0
        if probability <=1:
            if np.random.binomial(1, probability) is 1:
                keys_list.append(key)
        else:
            repeat_nums = round(probability)
            for r_num in range(int(repeat_nums)):
                keys_list.append(key)
    return keys_list


# 没有大于1的概率
def data_balance(voxel_map, gt_map, label_probability):
    keys = common.get_common_keys(voxel_map, gt_map)
    voxel_res = dict()
    gt_res = dict()

    near_array = search_kd_tree(keys, NEAR_NUM, MAX_DISTANCE)
    for i in range(len(keys)):
        voxel_info = list()
        key = keys[i]
        label = int(gt_map[key].feature_info_list[0].feature_list[0])
        near_keys = near_array[i, :, :]
        if label < common.class_num and label >= 0:
            probability = label_probability[label]
        else:
            probability = 0
        if np.random.binomial(1, probability) is 1:
            for j in range(near_keys.shape[0]):
                if near_keys[j, 0] != 0 and near_keys[j,1] != 0 and near_keys[j,2]!=0:
                    voxel_info.append(voxel_map[tuple(near_keys[j,:])])
            voxel_res[key] = voxel_info
            gt_res[key] = gt_map[key]
    return voxel_res, gt_res


# 对某些较少的类别进行强制重复
def data_balance_new(voxel_map, gt_map, label_probability):   #add repeat data when p >1
    keys = common.get_common_keys(voxel_map, gt_map)
    voxel_res = dict()
    gt_res = dict()

    keys_list = list()
    near_array = search_kd_tree(keys, NEAR_NUM, MAX_DISTANCE)
    for i in range(len(keys)):
        voxel_info = list()
        key = keys[i]
        label = int(gt_map[key].feature_info_list[0].feature_list[0])
        near_keys = near_array[i, :, :]
        probability = label_probability[label]
        if probability <=1:
            if np.random.binomial(1, probability) is 1:
                for j in range(near_keys.shape[0]):
                    if near_keys[j, 0] != 0 and near_keys[j,1] != 0 and near_keys[j,2]!=0:
                        voxel_info.append(voxel_map[tuple(near_keys[j,:])])
                voxel_res[key] = voxel_info
                gt_res[key] = gt_map[key]
                keys_list.append(key)
        else:
            for j in range(near_keys.shape[0]):
                if near_keys[j, 0] != 0 and near_keys[j,1] != 0 and near_keys[j,2]!=0:
                    voxel_info.append(voxel_map[tuple(near_keys[j,:])])
            # new_key = tuple(list(key) + [r_num])
            voxel_res[key] = voxel_info
            gt_res[key] = gt_map[key]

            repeat_nums = round(probability)
            for r_num in range(int(repeat_nums)):
                keys_list.append(key)

    return voxel_res, gt_res, keys_list


def search_kd_tree(keys, num, max_dist):
    near_array = np.zeros((len(keys), num, 3), dtype=int)
    kdtree = KDTree(keys, leaf_size=num)
    dist, ind = kdtree.query(keys, k=num)
    shape = dist.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (dist[i][j] <= max_dist):
                near_array[i][j] = keys[ind[i][j]]
            else:
                break
    return near_array


# output: full file path list
def get_file_list(data_dir):
    path_list = list()
    for i in os.listdir(data_dir):
        path_list.append(os.path.join(data_dir, i))
    return path_list


def make_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


if __name__ == "__main__":
    root_path = '/home/wangkai/project2/RnnFusion/data/CARLA_episode_0019/test3/'
    infer_block_file_path = root_path + 'infer_feature/'
    gt_block_file_path = root_path + 'gt_feature'
    infer_save_path = root_path + 'infer_feature_balance/'
    gt_save_path = root_path + 'gt_feature_balance/'
    make_path(infer_save_path)
    make_path(gt_save_path)

    infer_file = get_file_list(infer_block_file_path)
    infer_file.sort()
    gt_file = get_file_list(gt_block_file_path)
    gt_file.sort()
    np.random.seed(10)
    label = np.loadtxt('./data_dropout_ratio.txt')
    #label = np.random.rand(common.class_num)

    for i in range(len(infer_file)):
        i = 145
        infer_filename = infer_file[i]
        gt_filename = gt_file[i]

        infer_dict = np.load(infer_filename).item()
        gt_dict = np.load(gt_filename).item()
        voxel_res, gt_res = data_balance(infer_dict, gt_dict, label)

        infer_filename = infer_filename.split('/')[-1]
        gt_filename = gt_filename.split('/')[-1]
        print('ok!')
        np.save(infer_save_path + infer_filename, voxel_res)
        np.save(gt_save_path + gt_filename, gt_res)


