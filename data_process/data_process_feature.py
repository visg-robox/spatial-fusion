"""
    provide data for network
    copyright: zhang jian
"""
import time
import numpy
from data_structure.voxel_map import *
from data_structure.voxel_feature import *
from sklearn.neighbors import KDTree

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


def file_to_voxelmap(file_name, voxel_map, pose):
    start_time = time.time()
    fea_data = read_pointcloud_feature_npy(file_name)
    print(file_name)
    for i in range(len(fea_data)):
        voxel_center = voxel_regular(fea_data[i].location)
        vector = cal_vector(pose, voxel_center)
        feature_info = FeatureInfo(fea_data[i].feature_list, vector)
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


def pre_process(data_path,save_path, pose_path):
    point_file_list = get_file_list(data_path)
    point_file_list.sort()
    pose_file_list = get_file_list(pose_path)
    pose_file_list.sort()
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    pose_initial = read_pose(pose_file_list[0])
    file_map = VoxelMap(pose_initial)
    for i in range(len(point_file_list)):
        pose = read_pose(pose_file_list[i])
        file_to_voxelmap(point_file_list[i], file_map, pose)
        file_map.move(pose, save_path)
    file_map.unload_map(save_path)


TRAIN_FLAG = True
TEST_FLAG = False


if __name__ == '__main__':

    if TRAIN_FLAG is True:
        data_path = '/media/luo/Dataset/RnnFusion/apollo_data/Record067/Camera 6'
        feature_path = data_path + '/infer_feature/'
        p_path = data_path + '/infer/'
        gt_path = data_path + '/gt/'
        pose_path = data_path + '/pose/'
        save_path = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data'
        feature_save_path = save_path + '/infer_feature/'
        p_save_path = save_path + '/infer/'
        gt_save_path = save_path + '/gt_feature/'
        pre_process(gt_path, gt_save_path, pose_path)
        pre_process(feature_path, feature_save_path, pose_path)
        pre_process(p_path, p_save_path, pose_path)


    if TEST_FLAG is True:
        data_path = '/home/zhangjian/code/project/data/CARLA_episode_0019/'
        infer_path = data_path + 'test1/test/infer_feature/'
        gt_path = data_path + 'test1/test/gt_feature/'
        pose_path = data_path + 'test1/test/pose/'
        infer_save_path = data_path + 'test2/test_feature/infer/'
        gt_save_path = data_path + 'test2/test_feature/gt/'
        pre_process(gt_path, gt_save_path, pose_path)