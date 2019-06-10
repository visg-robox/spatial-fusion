import sys
sys.path.append("../")
import common
from torch import nn
import torch
from random import choice
from model.rnn import *
from torch.autograd import Variable
from data_process.data_process_label import *
from data_process import data_loader_torch,data_balance


BATCH_SIZE = common.batch_size
#这里用来修改可视化的中心坐标以及范围
visionlize_center = [-12142.5, 3250.5, 40]
Range = np.array([common.region_x, common.region_y, common.region_z]) * common.block_len


# input: data file path, result file path, fusion method
# output: write .txt file
def visualize_pc(data_path, gt_path = None, save_path = '.', fusion_method = 0, model_path=None):
    if model_path:
        rnn = torch.load(model_path)
        rnn.cuda()
        rnn.eval()
    result_map = VoxelMap(visionlize_center)
    map_name = common.FsMethod(fusion_method).name + '_visualization'

    infer_file_list = common.get_file_list_with_pattern('infer_feature', data_path)
    gt_file_list = common.get_file_list_with_pattern('gt', data_path)

    for num, source_filename in enumerate(infer_file_list):
        position = source_filename.split('/')[-1].split('.')[0].split('_')
        position = np.array(position,dtype=np.int)
        distance = abs(np.array(visionlize_center) - position/100)
        if not np.greater_equal(distance, Range).any():
            voxel_dict = np.load(source_filename).item()
            keys_list = list(voxel_dict.keys())
            print(len(keys_list))
            if gt_path:
                gt_filename = gt_file_list[num]
                gt_dict = np.load(gt_filename).item()
                gt = data_loader_torch.featuremap_to_gt_num(gt_dict,
                                                                keys_list,
                                                                len(keys_list),
                                                                ignore_list=common.ignore_list)

                valid_index = np.where(np.logical_and(np.greater_equal(gt, 0), np.less_equal(gt, common.class_num)))

                keys_list = list(np.array(keys_list)[valid_index])
                keys_list = list(map(lambda a:tuple(a), keys_list))
                gt = gt[valid_index]
                print(len(keys_list))

            print('source file name: ', source_filename)
            if fusion_method is common.FsMethod.STF:
                label_p = np.ones(common.class_num)
                infer_dict_res, gt_dict_res = data_balance.data_balance(voxel_dict, gt_dict, label_p)
                batch_num = (len(keys_list) // BATCH_SIZE) + 1
                for i in range(batch_num):
                    start_time = time.time()
                    if i == batch_num - 1:
                        current_keys = keys_list[i * BATCH_SIZE:]
                    else:
                        current_keys = keys_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

                    input_data = data_loader_torch.featuremap_to_batch_ivo_with_neighbour(infer_dict_res,
                                                                        current_keys,
                                                                        BATCH_SIZE,
                                                                        common.near_num,
                                                                        common.time_step,
                                                                        common.feature_num_ivo)
                    input_data = Variable(input_data).cuda()
                    with torch.no_grad():
                        output = rnn(input_data)
                    for j in range(len(current_keys)):
                        output_cpu = output.cpu().data.numpy()
                        result_map.insert(key_to_center(current_keys[j]), output_cpu[j])

            if fusion_method is common.FsMethod.RNN_FEATURE:
                batch_num = (len(keys_list) // BATCH_SIZE) + 1
                for i in range(batch_num):
                    start_time = time.time()
                    if i == batch_num - 1:
                        current_keys = keys_list[i * BATCH_SIZE:]
                    else:
                        current_keys = keys_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

                    input_data = data_loader_torch.featuremap_to_batch_i(voxel_dict,
                                                                                    current_keys,
                                                                                    BATCH_SIZE,
                                                                                    common.time_step,
                                                                                    common.img_feature_size)
                    input_data = Variable(input_data).cuda()
                    with torch.no_grad():
                        output = rnn(input_data,common.time_step)
                    for j in range(len(current_keys)):
                        output_cpu = output.cpu().data.numpy()
                        result_map.insert(key_to_center(current_keys[j]), output_cpu[j])

            if fusion_method is common.FsMethod.GT:
                for j in range(len(keys_list)):
                    output =  gt.cpu().data.numpy()
                    result_map.insert(key_to_center(keys_list[j]), output[j])

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
                    infer_label = choice(cur_voxel.feature_info_list).feature_list
                    result_map.insert(key_to_center(key), infer_label)

    if fusion_method is common.FsMethod.GT:
        result_map.write_map(save_path, map_name, onehot = False)
    else:
        result_map.write_map(save_path, map_name, onehot=True)


# True, False
VISUALIZE_STF = True
VISUALIZE_LSTM_FEATURE = False
VISUALIZE_RNN_LABEL   = False
VISUALIZE_BAYES       = False
VISUALIZE_BASELINE    = False
VISUALIZE_GT = False

if __name__ == '__main__':
    if VISUALIZE_STF:

        model_path = common.visualize_model
        save_path = os.path.join(os.path.dirname(model_path), common.model_step + '_visualization')
        test_path = os.path.join(common.blockfile_path, 'test')
        visualize_pc(test_path, save_path, common.FsMethod.STF, model_path)

    if VISUALIZE_LSTM_FEATURE:
        root_path = common.project_path
        model_path = root_path + 'train/feature/runs/LSTM_imgfeature/47500_model.pkl'
        save_path = root_path + 'train/feature/runs/LSTM_imgfeature/'
        data_path = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/test_feature/infer'
        gt_path = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/test_feature/gt'
        visualize_pc(data_path, gt_path, save_path, common.FsMethod.RNN_FEATURE, model_path)

    if VISUALIZE_BAYES:
        save_path = '/home/zhangjian/code/project/RnnFusion/record/bayes/'
        data_path = '/home/zhangjian/code/data/CARLA_episode_0019/test2/test1/infer'
        visualize_pc(data_path, save_path, common.FsMethod.BAYES)


    if VISUALIZE_BASELINE:
        data_path = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/test_feature/infer_label'
        gt_path = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/test_feature/gt'
        save_path = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/test_feature/'
        visualize_pc(data_path, gt_path, save_path, common.FsMethod.BASELINE)

    if VISUALIZE_GT:
        root_path = common.project_path
        save_path  = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/test_feature/'
        data_path  = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/test_feature/gt'
        visualize_pc(data_path, data_path, save_path, common.FsMethod.GT)