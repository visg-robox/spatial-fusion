# import common
from torch import nn
import torch
from random import choice
from model.rnn import *
from torch.autograd import Variable
from data_process.data_process import *
from data_process import data_loader_torch


# input: data file path, result file path, fusion method
# output: write .txt file
def visualize_pc(data_path, save_path, fusion_method, model_path=None):
    if fusion_method is common.FsMethod.RNN_FEATURE:
        # rnn = SSNet(common.feature_num, common.feature_num, common.class_num)
        rnn = torch.load(model_path)
    result_map = VoxelMap([-140, 120, 40])
    map_name = common.FsMethod(fusion_method).name + '_res'
    data_source_path = get_file_list(data_path)
    data_source_path.sort()

    for file_num in range(len(data_source_path)):
        source_filename = data_source_path[file_num]
        voxel_dict = np.load(source_filename).item()
        keys_list = list(voxel_dict.keys())
        print('source file name: ', source_filename)

        if fusion_method is common.FsMethod.RNN_FEATURE:
            for i in range(len(keys_list) // common.batch_size):
                start_time = time.time()
                current_keys = keys_list[i * common.batch_size:(i + 1) * common.batch_size]

                input_data = data_loader_torch.featuremap_to_batch(voxel_dict, current_keys, common.batch_size, common.time_step, common.feature_num)
                input_data = Variable(input_data, requires_grad=True).cuda()

                output = rnn(input_data, common.time_step)
                for j in range(len(current_keys)):
                    output_cpu = output.cpu().data.numpy()
                    result_map.insert(key_to_center(current_keys[j]), output_cpu[j])

        if fusion_method is common.FsMethod.BAYES:
            for key in keys_list:
                cur_voxel = voxel_dict[key]
                infer_label = [1 for _ in range(common.class_num)]
                for idx in range(len(cur_voxel.semantic_info_list)):
                    infer_label = [a * b for a, b in zip(infer_label, cur_voxel.semantic_info_list[idx].label_list)]
                result_map.insert(key_to_center(key), infer_label)

        if fusion_method is common.FsMethod.BASELINE:
            for key in keys_list:
                cur_voxel = voxel_dict[key]
                infer_label = choice(cur_voxel.semantic_info_list).label_list
                result_map.insert(key_to_center(key), infer_label)

    result_map.write_map(save_path, map_name)


# True, False
VISUALIZE_RNN_FEATURE = True
VISUALIZE_RNN_LABEL   = False
VISUALIZE_BAYES       = False
VISUALIZE_BASELINE    = False

if __name__ == '__main__':
    if VISUALIZE_RNN_FEATURE:
        root_path = common.project_path
        model_path = root_path + 'train/feature/59000_model.pkl'
        save_path  = root_path + 'train/feature/'
        data_path  = root_path + 'data/CARLA_episode_0019/test2/test_feature/infer'
        visualize_pc(data_path, save_path, common.FsMethod.RNN_FEATURE, model_path)
    if VISUALIZE_BAYES:
        save_path = '/home/zhangjian/code/project/RnnFusion/record/bayes/'
        data_path = '/home/zhangjian/code/data/CARLA_episode_0019/test2/test1/infer'
        visualize_pc(data_path, save_path, common.FsMethod.BAYES)
    if VISUALIZE_BASELINE:
        save_path = '/home/zhangjian/code/project/RnnFusion/record/icnet/'
        data_path = '/home/zhangjian/code/data/CARLA_episode_0019/test2/test1/infer'
        visualize_pc(data_path, save_path, common.FsMethod.BASELINE)
