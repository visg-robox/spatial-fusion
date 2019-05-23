# import common
from torch import nn
import torch
from random import choice
from model.rnn import *
from torch.autograd import Variable
from data_process.data_process_label import *
from data_process import data_loader_torch,data_balance

BATCH_SIZE = 32
#这里用来修改可视化的中心坐标以及范围
visionlize_center = [-12042.5, 2777.5, 42.5]
Range = np.array([common.region_x, common.region_y, common.region_z]) * common.block_len


# input: data file path, result file path, fusion method
# output: write .txt file
def visualize_pc(data_path, gt_path = None, save_path = '.', fusion_method = 0, model_path=None):
    if model_path:
        # rnn = SSNet(common.feature_num, common.feature_num, common.class_num)
        rnn = torch.load(model_path)
        rnn.cuda()
        rnn.eval()
    result_map = VoxelMap(visionlize_center)
    map_name = common.FsMethod(fusion_method).name + '_res'
    data_source_path = get_file_list(data_path)
    data_source_path.sort()
    gt_source_path = get_file_list(data_path)
    gt_source_path.sort()

    for num,source_filename in enumerate(data_source_path):
        position = source_filename.split('/')[-1].split('.')[0].split('_')
        position = np.array(position,dtype=np.int)
        distance = abs(np.array(visionlize_center) - position/100)
        if not np.greater_equal(distance, Range).any():
            voxel_dict = np.load(source_filename).item()
            keys_list = list(voxel_dict.keys())
            print(len(keys_list))
            if gt_path:
                gt_filename = gt_source_path[num]
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
                batch_num = (len(keys_list) // common.batch_size) + 1
                for i in range(batch_num):
                    start_time = time.time()
                    if i == batch_num - 1:
                        current_keys = keys_list[i * common.batch_size:]
                    else:
                        current_keys = keys_list[i * common.batch_size:(i + 1) * common.batch_size]

                    input_data = data_loader_torch.featuremap_to_batch_with_balance(infer_dict_res,
                                                                        current_keys,
                                                                        common.batch_size,
                                                                        common.near_num,
                                                                        common.time_step,
                                                                        common.feature_num)
                    input_data = Variable(input_data).cuda()
                    with torch.no_grad():
                        output = rnn(input_data)
                    for j in range(len(current_keys)):
                        output_cpu = output.cpu().data.numpy()
                        result_map.insert(key_to_center(current_keys[j]), output_cpu[j])

            if fusion_method is common.FsMethod.RNN_FEATURE:
                batch_num = (len(keys_list) // common.batch_size) + 1
                for i in range(batch_num):
                    start_time = time.time()
                    if i == batch_num - 1:
                        current_keys = keys_list[i * common.batch_size:]
                    else:
                        current_keys = keys_list[i * common.batch_size:(i + 1) * common.batch_size]

                    input_data = data_loader_torch.featuremap_to_batch(voxel_dict,
                                                                                    current_keys,
                                                                                    common.batch_size,
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
                    infer_label = choice(cur_voxel.semantic_info_list).label_list
                    result_map.insert(key_to_center(key), infer_label)

    if fusion_method is common.FsMethod.GT:
        result_map.write_map(save_path, map_name, onehot = False)
    else:
        result_map.write_map(save_path, map_name, onehot=True)


# True, False
VISUALIZE_STF = False
VISUALIZE_RNN_FEATURE = False
VISUALIZE_RNN_LABEL   = False
VISUALIZE_BAYES       = False
VISUALIZE_BASELINE    = False
VISUALIZE_GT = True

if __name__ == '__main__':
    if VISUALIZE_STF:
        root_path = common.project_path
        model_path = root_path + 'train/feature/runs/average_feature_new/15000newnew_model.pkl'
        save_path  = root_path + 'train/feature/runs/average_feature_new/'
        data_path  = '/media/luo/Dataset/RnnFusion/CARLA_episode_0019/test3/infer_feature'
        visualize_pc(data_path, save_path, common.FsMethod.STF, model_path)

    if VISUALIZE_RNN_FEATURE:
        root_path = common.project_path
        model_path = root_path + 'train/feature/runs/lstm_no_vector/50000_model.pkl'
        save_path = root_path + 'train/feature/runs/lstm_no_vector/'
        data_path = '/media/luo/Dataset/RnnFusion/CARLA_episode_0019/test3/infer_feature'
        visualize_pc(data_path, save_path, common.FsMethod.RNN_FEATURE, model_path)

    if VISUALIZE_BAYES:
        save_path = '/home/zhangjian/code/project/RnnFusion/record/bayes/'
        data_path = '/home/zhangjian/code/data/CARLA_episode_0019/test2/test1/infer'
        visualize_pc(data_path, save_path, common.FsMethod.BAYES)


    if VISUALIZE_BASELINE:
        save_path = '/home/zhangjian/code/project/RnnFusion/record/icnet/'
        data_path = '/home/zhangjian/code/data/CARLA_episode_0019/test2/test1/infer'
        visualize_pc(data_path, save_path, common.FsMethod.BASELINE)

    if VISUALIZE_GT:
        root_path = common.project_path
        save_path  = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data'
        data_path  = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/gt_feature'
        visualize_pc(data_path, data_path, save_path, common.FsMethod.GT)