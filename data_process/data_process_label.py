"""
    provide data for network
    copyright: zhang jian
"""
import os
import time
import numpy
from data_structure.voxel_semantic import *
from data_structure.voxel_map import *


# output: full file path list



def read_pose(file_name):
    return np.loadtxt(file_name)[:-1, -1]


# read semantic point from .ply file
# file structure: x, y, z, label_1, ... , label_n
# output: semantic point list
def read_pointcloud_seg(file_name):
    point_list = []
    count = 0
    with open(file_name) as f:
        line = f.readline(7)
        while line:
            count += 1
            line = f.readline()
            if line is '':
                continue
            if count >= 7:
                line_list = line.split()
                line_list = [float(i) for i in line_list]
                semantic_point = SemanticLidarPoint(line_list[0:3], line_list[3:])
                point_list.append(semantic_point)
    return point_list


# read from npy
# output: semantic point list
def read_pointcloud_seg_npy(file_name):
    data = numpy.load(file_name)
    point_list = []
    for i in range(len(data)):
        semantic_point = SemanticLidarPoint(data[i][0:3], data[i][3:])
        point_list.append(semantic_point)
    return point_list


# read file, insert semantic points to voxel map
def file_to_voxelmap(file_name, voxel_map):
    start_time = time.time()
    seg_data = read_pointcloud_seg_npy(file_name)
    print(file_name)
    for i in range(len(seg_data)):
        voxel_center = voxel_regular(seg_data[i].location)
        voxel_info = SemanticInfo(seg_data[i].label_list)
        if voxel_map.find_location(voxel_center) is None:
            current_voxel = SemanticVoxel(voxel_center)
            current_voxel.insert_label(voxel_info)
            voxel_map.insert(voxel_center, current_voxel)
        else:
            # print('here')
            voxel_map.find_location(voxel_center).insert_label(voxel_info)
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
        # file_to_voxelmap(point_file_list[i], infer_map)
        infer_map.move(pose, infer_save_path)
    infer_map.unload_map(infer_save_path)

    gt_map = VoxelMap(pose_initial)
    for i in range(len(gt_file_list)):
        pose = read_pose(pose_file_list[i])
        file_to_voxelmap(gt_file_list[i], gt_map)
        gt_map.move(pose, gt_save_path)
    gt_map.unload_map(gt_save_path)


def make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

TRAIN_FLAG = True
TEST_FLAG = False


if __name__ == '__main__':
    if TRAIN_FLAG is True:
        data_path = '/home/zhangjian/code/data/CARLA_episode_0019/'
        infer_path = data_path + 'test1/infer/'
        gt_path = data_path + 'test1/gt/'
        pose_path = data_path + 'test1/infer_pose/'
        infer_save_path = data_path + 'test2/infer/'
        gt_save_path = data_path + 'test2/gt/'
        pre_process(infer_path, gt_path, pose_path, infer_save_path, gt_save_path)

    if TEST_FLAG is True:
        root_path = '/home/zhangjian/code/data/CARLA_episode_0019/'
        infer_path = root_path + 'test1/test/infer/'
        gt_path = root_path + 'test1/test/gt/'
        pose_path = root_path + 'test1/test/pose/'
        infer_save_path = root_path + 'test2/test/infer/'
        gt_save_path = root_path + 'test2/test/gt/'
        pre_process(infer_path, gt_path, pose_path, infer_save_path, gt_save_path)


