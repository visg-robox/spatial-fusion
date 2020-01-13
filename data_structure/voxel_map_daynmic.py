
"""
duty:
1. maintain 3d structure
2. provide & update feature info batch
    1) lidar point input
    2) related block
    3) feature info batch
    4) network output
    5) update voxel
    6) save to file
"""


import os
import common
import random
import shelve
import numpy as np
from data_structure.voxel_feature import *
# from model_evaluate.eval_semantic_apollo import ID_COLOR


class VoxelMap:

    def __init__(self, batch_size, path):
        self.map = dict()
        self.batch_size = batch_size
        self.path = path

    def reset(self):
        self.map = dict()

    # load map, if related blocks do not exist, create it
    # find voxel, if related voxel do not exist, create it
    def get_feature_batch(self, batch_points):
        batch_size = batch_points.shape[0]
        hidden_feature_batch = np.zeros((batch_size, common.feature_num_ivo))
        block_dict = get_related_blocks(batch_points)
        block_keys = list(block_dict.keys())
        self.load_map(block_keys)
        for i in range(len(batch_points)):
            point  = batch_points[i]
            voxel = self.find_voxel(point)
            if voxel is not None:
                hidden_feature_batch[i] = voxel.feature
        return hidden_feature_batch

    # update related voxels according to points
    def update_feature(self, batch_points, batch_features):
        for i in range(len(batch_points)):
            point = batch_points[i]
            voxel = self.find_voxel(point)
            voxel.update(batch_features[i])

    def unload_map(self, path):
        for i in range(len(self.map)):
            self.map[i].close()

    # input: list of block center in form of tuple, which is dict key
    # operation: load related block to self.map
    def load_map(self, block_keys):
        for i in range(len(block_keys)):
            self.load_block(block_keys[i])

    # location operation, data can be semantic voxel or feature voxel
    def insert_voxel(self, location, data):
        current_block = self.find_block(location)
        voxel_center = voxel_regular(location)
        voxel_key = voxel_center_to_string(voxel_center)
        current_block[voxel_key] = data

    def delete_voxel(self, location):
        current_block = self.find_block(location)
        voxel_center = voxel_regular(location)
        voxel_key = voxel_center_to_string(voxel_center)
        current_block.pop(voxel_key)

    # arbitrary location in voxel
    # output: feature voxel
    def find_voxel(self, location):
        current_block = self.find_block(location)
        voxel_center = voxel_regular(location)
        voxel_key = voxel_center_to_string(voxel_center)
        if voxel_key in current_block:
            voxel = current_block[voxel_key]
            return voxel
        else:
            current_block[voxel_key] = FeatureVoxel(voxel_center)

    def update_voxel(self, location, feature):
        voxel = self.find_voxel(location)
        voxel.update(feature)

    # according to current lidar points, load related blocks
    def load_related_blocks(self, blocks_dict):
        keys_list = blocks_dict.keys()
        self.load_map(keys_list)

    def find_block(self, location):
        block_center = block_regular(location)
        current_block = self.map[tuple(block_center)]
        return current_block

    def load_block(self, location):
        block_center = block_regular(location)
        block_name = self.path + (''.join((str(int(e * 100)) + '_') for e in block_center.tolist()))[:-1] + '.db'
        block_db = shelve.open(block_name, flag='c', protocol=2, writeback=True)
        self.map[tuple(block_center)] = block_db

    def keys(self, idx):
        return self.map.keys()


# block_regular to block center
# input: arbitrary location in block
# output: block center
def block_regular(location):
    location = np.array(location)
    block_center = (location // common.block_len + 0.5) * common.block_len
    return np.round(block_center, 3)


# input: arbitrary location in voxel
# output: voxel center
def voxel_regular(location):
    location = np.array(location)
    voxel_center = (location // common.voxel_length + 0.5) * common.voxel_length
    return np.round(voxel_center, 3)


# because shelve module require that key should be string
def voxel_center_to_string(voxel_center):
    voxel_key = (''.join((str(int(e * 1000)) + '_') for e in voxel_center.tolist()))[:-1]
    return voxel_key


# arbitrary location to key
def center_to_key(location):
    voxel_key = (voxel_regular(location)/common.voxel_length * 2).round().astype(np.int32)
    return tuple(voxel_key)


# output: location in real word
def key_to_center(voxel_key):
    voxel_center = np.array(voxel_key)/2 * common.voxel_length
    return voxel_center


def find_file(start, name):
    full_path = None
    for relpath, dirs, files in os.walk(start):
        if name in files:
            full_path = os.path.join(start, relpath, name)
    return full_path


def point_to_batches(point_list, batch_size):
    voxel_list = []
    for point in point_list:
        voxel_center = voxel_regular(point)
        if not voxel_center in voxel_list:
            voxel_list.append(voxel_center)
    batches = []
    random.shuffle(voxel_list)
    for i in range(len(voxel_list)//batch_size):
        batches.append(np.array(voxel_list[i*batch_size:(i+1)*batch_size]))
    return batches


# input: should be shuffled, batch sized lidar points
# related blocks center/key
def get_related_blocks(batch_points):
    blocks_dict = dict()
    for i in range(len(batch_points)):
        block_center = block_regular(batch_points[i])
        blocks_dict.setdefault(tuple(block_center), []).append(batch_points[i])
    return blocks_dict


def read_ply(file_name):
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
                point = [float(i) for i in line_list]
                point_list.append(point)
    return point_list

 
if __name__ == '__main__':
    point_list = read_ply('/home/zhuxiliuyun/code/project/data/image_lidar/Lidar/Lidar64_1/000000.ply')
    point_batches = point_to_batches(point_list, 1000)
    thd_info_path = '/home/zhuxiliuyun/code/project/data/3d_info/'
    map_3d = VoxelMap(1000, thd_info_path)
    hidden_feature = map_3d.get_feature_batch(point_batches[0])
"""
    output_labels, output_feature = network(hidden_feature, input_feature)
    loss = loss_function(output_labels, gt_labels)
    network.back_propagation(loss, parameter)
    map_3d.update_feature(output_feature, point_batches[0])
"""

