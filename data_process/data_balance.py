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


def data_balance(voxel_map, gt_map, label_probability):
    voxel_keys = list(voxel_map.keys())
    gt_keys = list(gt_map.keys())
    keys = [v for v in voxel_keys if v in gt_keys]

    voxel_res = dict()
    gt_res = dict()

    near_array = search_kd_tree(keys, NEAR_NUM, MAX_DISTANCE)
    for i in range(len(keys)):
        voxel_info = list()
        key = keys[i]
        label = int(gt_map[key].feature_info_list[0].feature_list[0])
        near_keys = near_array[i, :, :]
        probability = label_probability[label]
        if np.random.binomial(1, probability) is 1:
            for i in range(near_keys.shape[0]):
                if near_keys[i, 0]+near_keys[i, 1]+near_keys[i, 2] is not 0:
                    voxel_info.append(voxel_map[tuple(near_keys[i,:])])
            voxel_res[key] = voxel_info
            gt_res[key] = gt_map[key]
    return voxel_res, gt_res


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
    root_path = '/home/zhangjian/code/project/data/CARLA_episode_0019/test3/'
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
    label = np.random.rand(common.class_num)

    for i in range(len(infer_file)):
        infer_filename = infer_file[i]
        gt_filename = gt_file[i]

        infer_dict = np.load(infer_filename).item()
        gt_dict = np.load(gt_filename).item()
        voxel_res, gt_res = data_balance(infer_dict, gt_dict, label)

        infer_filename = infer_filename.split('/')[-1]
        gt_filename = gt_filename.split('/')[-1]
        np.save(infer_save_path + infer_filename, voxel_res)
        np.save(gt_save_path + gt_filename, gt_res)





