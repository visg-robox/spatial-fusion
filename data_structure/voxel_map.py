import os
import common
import numpy as np
from model_evaluate.eval_API import ID_COLOR


class VoxelMap:

    def __init__(self, location):
        self.center = block_regular(location)
        self.map = list()
        for i in range(common.region_x*common.region_y*common.region_z):
            self.map.append(dict())

    def move(self, location, save_path):
        if self.region_control(location):
            new_map = VoxelMap(block_regular(location))
            map_start = new_map.get_map_start()
            map_end = new_map.get_map_end()
            for i in range(common.region_x * common.region_y * common.region_z):
                index_3d = idx_to_center(i)
                block_center = self.idx2_block_center(index_3d)
                if ((block_center - map_start) > 0).all() and ((map_end - block_center) > 0).all():
                    new_map.map[center_to_idx(new_map.get_block_center(block_center))] = self.map[i]
                else:
                    self.unload_block(block_center, save_path)
            self.reset()
            self.center = new_map.center
            self.map = new_map.map

    def reset(self):
        self.center = np.array([0, 0, 0])
        self.map = list()

    def region_control(self, location):
        if (abs(self.center - location) > common.block_len).any():
            return True
        return False

    def unload_map(self, path):
        for i in range(len(self.map)):
            center_index = idx_to_center(i)
            block_location = self.idx2_block_center(center_index).astype(np.int32)
            block_name = path + '/' + (''.join((str(int(e * 100)) + '_') for e in block_location.tolist()))[:-1] + '.npy'
            if len(self.map[i]) > 100:
                np.save(block_name, self.map[i])
            self.map[i].clear()

    def get_map_start(self):
        map_start = self.center - np.array([common.region_x*common.block_len/2,
                                            common.region_y*common.block_len/2,
                                            common.region_z*common.block_len/2])
        return map_start

    def get_map_end(self):
        map_end = self.center + np.array([common.region_x*common.block_len/2,
                                            common.region_y*common.block_len/2,
                                            common.region_z*common.block_len/2])
        return map_end

    # block 3d index to block center
    def idx2_block_center(self, index_3d):
        map_start = self.get_map_start()
        block_center = map_start + (np.array(index_3d) + 0.5) * common.block_len
        return block_center

    # convert arbitrary location to block index
    # input: arbitrary location in map
    # output: 3d array index of block
    def get_block_center(self, location):
        map_start = self.get_map_start()
        return (location - map_start)//common.block_len

    # location operation, data can be semantic voxel or feature voxel
    def insert(self, location, data):
        if self.point_in_range(location) is True:
            block_idx = center_to_idx(self.get_block_center(location))
            voxel_key = center_to_key(location)
            self.map[block_idx][voxel_key] = data

    def delete(self, location):
        block_idx = center_to_idx(self.get_block_center(location))
        voxel_key = center_to_key(location)
        self.map[block_idx].pop(voxel_key)

    # arbitrary location in voxel
    def find_location(self, location):
        if self.point_in_range(location) is True:
            block_idx = center_to_idx(self.get_block_center(location))
            voxel_key = center_to_key(location)
            is_existed = self.map[block_idx].get(voxel_key)
            if is_existed is None:
                return None
            return self.map[block_idx][voxel_key]
        return None

    # find_key through voxel key
    def find_key(self, voxel_key):
        voxel_center = key_to_center(voxel_key)
        if self.point_in_range(voxel_center) is True:
            block_idx = center_to_idx(self.get_block_center(voxel_center))
            is_existed = self.map[block_idx].get(voxel_key)
            if is_existed is None:
                return None
            return self.map[block_idx][voxel_key]
        return None

    def set(self, location, value):
        block_idx = center_to_idx(self.get_block_center(location))
        voxel_key = center_to_key(location)
        self.map[block_idx][voxel_key] = value

    def keys(self, idx):
        return self.map[idx].keys()

    # arbitrary location in block
    def unload_block(self, location, path):
        block_idx = center_to_idx(self.get_block_center(location))
        block_name = path + '/' + (''.join((str(int(e*100)) + '_') for e in location.tolist()))[:-1] + '.npy'
        # if len(self.map[block_idx]) < common.batch_size:
        #     self.map[block_idx].clear()
        if len(self.map[block_idx]) > common.batch_size:
            np.save(block_name, self.map[block_idx])
        self.map[block_idx].clear()

    def load_block(self, location, path):
        block_idx = center_to_idx(self.get_block_center(location))
        block_center = block_regular(location)
        block_name = (''.join((str(int(e * 100)) + '_') for e in block_center.tolist()))[:-1] + '.npy'
        block_fullpath = find_file(path, block_name)
        if block_fullpath is None:
            return None
        voxel_map = np.load(block_fullpath).item()
        self.map[block_idx] = voxel_map

    # whether point in range
    def point_in_range(self, point_location):
        point_location = np.array(point_location)
        point_difference = np.abs(self.center - point_location)
        if point_difference[0] >= common.region_x * common.block_len / 2:
            return False
        if point_difference[1] >= common.region_y * common.block_len / 2:
            return False
        if point_difference[2] >= common.region_z * common.block_len / 2:
            return False
        return True

    def write_map(self, path, file_name, onehot = True):
        full_path = os.path.join(path, file_name)
        obj_file = open(full_path + '.txt', 'w')
        for i in range(len(self.map)):
            for key in self.map[i]:
                location = key_to_center(key)
                if  onehot:
                    index = np.argmax(self.map[i][key])
                else:

                    index =  self.map[i][key]
                    if index < 0:
                        index = 0
                print('index=', index)
                color = ID_COLOR[index]
                line = str(location[0]) + ' ' + str(location[1]) + ' ' + str(location[2]) + ' ' + \
                       str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + '\n'
                obj_file.write(line)
            self.map[i].clear()
        obj_file.close()


# block_regular to block center
# input: arbitrary location in block
# output: block center
def block_regular(location):
    location = np.array(location)
    block_center = (location // common.block_len + 0.5) * common.block_len
    return block_center


# input: arbitrary location in voxel
# output: voxel center
def voxel_regular(location):
    location = np.array(location , dtype= np.float32)
    voxel_center = (location // common.voxel_length + 0.5) * common.voxel_length
    return voxel_center


# 3d block center index to block index.  such as (0,0,0) to 0
def center_to_idx(block_center):
    if block_center[0] >= common.region_x or block_center[1] >= common.region_y or block_center[2] >= common.region_z:
        raise Exception('Invalid center idx')
    idx = block_center[0] + block_center[1] * common.region_x + block_center[2] * common.region_x * common.region_y
    return int(idx)


# block index to 3d block center index
def idx_to_center(index):
    center_idx = np.zeros(3).astype(np.int32)
    center_idx[2] = index // (common.region_x * common.region_y)
    center_idx[1] = (index - common.region_x * common.region_y * center_idx[2])//common.region_x
    center_idx[0] = index % common.region_x
    return center_idx


# arbitrary location to key
def center_to_key(location):
    voxel_key = (voxel_regular(location)/common.voxel_length * 2).round().astype(np.int32)
    return tuple(voxel_key)


# output: location in real word
def key_to_center(voxel_key):
    voxel_center = np.array(voxel_key)/ 2 * common.voxel_length
    return voxel_center


def find_file(start, name):
    full_path = None
    for relpath, dirs, files in os.walk(start):
        if name in files:
            full_path = os.path.join(start, relpath, name)
    return full_path


class VoxelBlock:
    pass







if __name__ == '__main__':
    key = center_to_key([-10.2, -1.2, -1.2])
    print(key)
    center = key_to_center(key)
    print(center)
    key = center_to_key(center)
    print(key)
    center = key_to_center(key)
    print(center)

    index = center_to_idx(np.array([5,2,3]))
    print(index)
    center = idx_to_center(index)
    print(center)

