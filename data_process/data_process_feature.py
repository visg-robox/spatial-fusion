"""
    provide data for network
    copyright: zhang jian
"""
import time
import numpy
from data_structure.voxel_map import *
from data_structure.voxel_feature import *
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


def file_to_voxelmap(file_name, voxel_map):
    start_time = time.time()
    fea_data = read_pointcloud_feature_npy(file_name)
    print(file_name)
    for i in range(len(fea_data)):
        voxel_center = voxel_regular(fea_data[i].location)
        feature_info = FeatureInfo(fea_data[i].feature_list)
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
        file_to_voxelmap(point_file_list[i], infer_map)
        infer_map.move(pose, infer_save_path)
    infer_map.unload_map(infer_save_path)

    gt_map = VoxelMap(pose_initial)
    for i in range(len(gt_file_list)):
        pose = read_pose(pose_file_list[i])
        file_to_voxelmap(gt_file_list[i], gt_map)
        gt_map.move(pose, gt_save_path)
    gt_map.unload_map(gt_save_path)


TRAIN_FLAG = True
TEST_FLAG = False


if __name__ == '__main__':

    if TRAIN_FLAG is True:
        root_path = '/home/zhangjian/code/data/CARLA_episode_0019/'
        infer_path = root_path + 'test1/infer_feature/'
        gt_path = root_path + 'test1/gt_feature/'
        pose_path = root_path + 'test1/pose/'
        infer_save_path = root_path + 'test2/infer_feature/'
        gt_save_path = root_path + 'test2/gt_feature/'
        pre_process(infer_path, gt_path, pose_path, infer_save_path, gt_save_path)

    if TEST_FLAG is True:
        root_path = '/home/zhangjian/code/data/CARLA_episode_0019/'
        infer_path = root_path + 'test1/test/infer_feature/'
        gt_path = root_path + 'test1/test/gt_feature/'
        pose_path = root_path + 'test1/test/pose/'
        infer_save_path = root_path + 'test2/test_feature/infer/'
        gt_save_path = root_path + 'test2/test_feature/gt/'
        pre_process(infer_path, gt_path, pose_path, infer_save_path, gt_save_path)