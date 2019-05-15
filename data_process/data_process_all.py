"""
    provide data for network
    copyright: zhang jian
"""
import time
import numpy
from data_structure.voxel_map import *
from data_structure.voxel_feature import *
from sklearn.neighbors import KDTree


near_num = 25
max_dist = 1
# from data_process.data_process import get_file_list, read_pose


# output: full file path list
def get_file_list(data_dir):
    path_list = list()
    for i in os.listdir(data_dir):
        path_list.append(os.path.join(data_dir, i))
    return path_list


def read_pose(file_name):
    return np.loadtxt(file_name)[:-1, -1]


def read_pointcloud_feature_npy(file_name):
    data = numpy.load(file_name)
    feature_list = []
    for i in range(len(data)):
        feature_point = FeatureLidarPoint(data[i][0:3], data[i][3:])
        feature_list.append(feature_point)
    return feature_list


def cal_vector(pose, voxel_idx):
    return [pose[i] - voxel_idx[i] for i in range(len(pose))]


def get_all_voxel_keys(fea_data):
    keys = np.zeros((len(fea_data), 3))
    for i in range(len(fea_data)):
        keys[i] = key_to_center(np.array(fea_data[i].location))

    return keys


def search_kd_tree(keys, num, max_dist):
    near_array = np.zeros((len(keys), num, 3))
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


def file_to_voxelmap(file_name, voxel_map, pose):
    start_time = time.time()
    fea_data = read_pointcloud_feature_npy(file_name)
    print(file_name)

    keys = get_all_voxel_keys(fea_data)
    neay_arrays = search_kd_tree(keys, near_num, max_dist)

    for i in range(len(fea_data)):
        voxel_center = voxel_regular(fea_data[i].location)
        vector = cal_vector(pose, voxel_center)
        feature_info = FeatureInfo_new(fea_data[i].feature_list, vector, neay_arrays[i])
        if voxel_map.find_location(voxel_center) is None:
            current_voxel = FeatureVoxel(voxel_center)
            current_voxel.insert_feature(feature_info)
            voxel_map.insert(voxel_center, current_voxel)
        else:
            # print('here')
            voxel_map.find_location(voxel_center).insert_feature(feature_info)
            # print(current_voxel)
    end_time = time.time()
    used_time = end_time - start_time
    print('This frame uses', used_time, 's')


def pre_process(infer_path, gt_path, pose_path, infer_save_path, gt_save_path):
    point_file_list = get_file_list(infer_path)
    point_file_list.sort()
    gt_file_list = get_file_list(gt_path)
    gt_file_list.sort()
    pose_file_list = get_file_list(pose_path)
    pose_file_list.sort()

    pose_initial = read_pose(pose_file_list[0])
    infer_map = VoxelMap(pose_initial)
    for i in range(len(point_file_list)):
        pose = read_pose(pose_file_list[i])
        file_to_voxelmap(point_file_list[i], infer_map, pose)
        infer_map.move(pose, infer_save_path)
    infer_map.unload_map(infer_save_path)

    gt_map = VoxelMap(pose_initial)
    for i in range(len(gt_file_list)):
        pose = read_pose(pose_file_list[i])
        file_to_voxelmap(gt_file_list[i], gt_map, pose)
        gt_map.move(pose, gt_save_path)
    gt_map.unload_map(gt_save_path)


TRAIN_FLAG = True
TEST_FLAG = False

if __name__ == '__main__':

    if TRAIN_FLAG is True:
        data_path = '/home/wangkai/project2/RnnFusion/data/CARLA_episode_0019/'
        infer_path = data_path + 'test1/infer_feature/'
        gt_path = data_path + 'test1/gt_feature/'
        pose_path = data_path + 'test1/pose/'
        infer_save_path = data_path + 'test4/infer_feature/'
        gt_save_path = data_path + 'test4/gt_feature/'
        pre_process(infer_path, gt_path, pose_path, infer_save_path, gt_save_path)

    if TEST_FLAG is True:
        data_path = '/home/zhangjian/code/project/data/CARLA_episode_0019/'
        infer_path = data_path + 'test1/test/infer_feature/'
        gt_path = data_path + 'test1/test/gt_feature/'
        pose_path = data_path + 'test1/test/pose/'
        infer_save_path = data_path + 'test2/test_feature/infer/'
        gt_save_path = data_path + 'test2/test_feature/gt/'
        pre_process(infer_path, gt_path, pose_path, infer_save_path, gt_save_path)