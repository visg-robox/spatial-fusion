"""
all rights reserved
"""

from data_process.data_process import *
from data_process import data_loader_torch
from matplotlib import pyplot as plt

BATCH_SIZE = common.batch_size
TIME_STEP = 100
INPUT_SIZE = 13


def batch_data_visualization(file_path):

    file_list = get_file_list(file_path)
    file_list.sort()

    for file_num in range(len(file_list)):
        infer_filename = file_list[file_num]
        voxel_dict = np.load(infer_filename).item()
        keys_list = list(voxel_dict.keys())
        print('infer file name: ', infer_filename)
        for i in range(len(keys_list) // BATCH_SIZE):
            current_keys = keys_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            # input_data shape: (BATCH_SIZE, TIME_STEP, INPUT_SIZE)
            input_data = data_loader_torch.labelmap_to_batch(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
            input_data = input_data.numpy()
            # just show first image in batch
            plt.imshow(input_data[0, :, :], cmap='gray')
            plt.show()
            print('picture:', i)


def visualize_batch(input_data):
    # input data shape: (time step, input_size)
    input_data = input_data.numpy()
    plt.imshow(input_data,  cmap='gray')
    plt.show()


if __name__ == '__main__':
    rootPath = '/home/zhangjian/code/project/RnnFusion/'
    inferPath = rootPath + 'data/CARLA_episode_0019/test2/infer/'
    # root_path = '/home/zhangjian/code/data/CARLA_episode_0019/test2/'
    # inferPath = root_path + 'infer/'
    batch_data_visualization(inferPath)



