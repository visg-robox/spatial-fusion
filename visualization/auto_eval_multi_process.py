import sys
sys.path.append('../')
import multiprocessing
import os
#from data_process.data_process_label import *
# import common
from torch import nn
import torch
from random import choice
from model.rnn import *
from torch.autograd import Variable
from data_process.data_process_label import *
from data_process import data_loader_torch,data_balance
import common
import shelve

BATCH_SIZE = 32



def visualize_pc(all_path, save_path = '.', fusion_method = 0, model_path=None):
    data_source_path = all_path[0]
    gt_source_path = all_path[1]
    if model_path:
        # rnn = SSNet(common.feature_num, common.feature_num, common.class_num)
        rnn = torch.load(model_path)
        rnn.cuda()
        rnn.eval()
    map_name = common.FsMethod(fusion_method).name + '_res'

    source_filename = data_source_path


    position = source_filename.split('/')[-1].split('.')[0].split('_')
    position = np.array(position,dtype=np.int)

    visionlize_center = [position[i]/100 for i in range(len(position))]
    result_map = VoxelMap(visionlize_center)

    voxel_dict = np.load(source_filename).item()
    keys_list = list(voxel_dict.keys())
    print(len(keys_list))
    if gt_path:
        gt_filename = gt_source_path
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

            input_data = data_loader_torch.featuremap_to_batch_iv(voxel_dict,
                                                                            current_keys,
                                                                            BATCH_SIZE,
                                                                            common.time_step,
                                                                            common.feature_num_iv)
            input_data = Variable(input_data).cuda()
            with torch.no_grad():
                output = rnn(input_data, common.time_step)
            for j in range(len(current_keys)):
                output_cpu = output.cpu().data.numpy()
                result_map.insert(key_to_center(current_keys[j]), output_cpu[j])

    if fusion_method is common.FsMethod.GT:
        for j in range(len(keys_list)):

            output = gt.cpu().data.numpy()
            result_map.insert(key_to_center(keys_list[j]), output[j])


    if fusion_method is common.FsMethod.BAYES:
        for key in keys_list:
            cur_voxel = voxel_dict[key]
            infer_label = [1 for _ in range(common.class_num)]
            for idx in range(len(cur_voxel.feature_info_list)):
                infer_label = [a * b for a, b in zip(infer_label, cur_voxel.feature_info_list[idx].feature_list)]
            result_map.insert(key_to_center(key), infer_label)

    if fusion_method is common.FsMethod.BASELINE:
        for key in keys_list:
            cur_voxel = voxel_dict[key]
            infer_label = choice(cur_voxel.feature_info_list).feature_list
            result_map.insert(key_to_center(key), infer_label)


    save_path_new = os.path.join(save_path,source_filename.split('/')[-3])
    make_dir(save_path_new)
    if fusion_method is common.FsMethod.GT:
        result_map.write_map(save_path_new, map_name+source_filename.split('/')[-1].split('.')[0], onehot = False)
    else:
        result_map.write_map(save_path_new, map_name+source_filename.split('/')[-1].split('.')[0], onehot=True)




def do(all_path):
    model_path = common.test_model_path + '/150000_model.pkl'
    save_path = os.path.join(common.test_model_path, 'visual_spnet_multiprocess')
    make_dir(save_path)

    visualize_pc(all_path, save_path, common.FsMethod.STF, model_path)

    return


if __name__ == '__main__' :
    time1 = time.time()
    pool = multiprocessing.Pool(processes=2)
    if common.para_dict['dataset_class_config'] == 'apollo':
        scene_name = 'Record006/'
        data_path = os.path.join(common.blockfile_path, 'test/' + scene_name + 'infer_feature/')
        gt_path = os.path.join(common.blockfile_path, 'test/' + scene_name + '/gt/')
        data_source_path = common.get_file_list(data_path)
        data_source_path.sort()
        gt_source_path = common.get_file_list(gt_path)
        gt_source_path.sort()
        length = len(data_source_path)

        all_paths = []
        for i in range(length):
            all_paths.append([data_source_path[i], gt_source_path[i]])

        pool.map(do, all_paths)
    if common.para_dict['dataset_class_config'] == 'S3DIS':
        list_path = sys.argv[4]
        room_list = []
        with open(list_path, 'r') as r_f:
            for line in r_f:
                room_list.append(line.strip())
        for item in room_list:
            data_path = os.path.join(common.blockfile_path, 'test/' + item + 'infer_feature/')
            gt_path = os.path.join(common.blockfile_path, 'test/' + item + 'gt/')
            data_source_path = common.get_file_list(data_path)
            data_source_path.sort()
            gt_source_path = common.get_file_list(gt_path)
            gt_source_path.sort()
            length = len(data_source_path)

            all_paths = []
            for i in range(length):
                all_paths.append([data_source_path[i], gt_source_path[i]])

            pool.map(do, all_paths)


    pool.close()
    pool.join()
    time2 = time.time()
    print("Sub-process(es) done.")
    print(time2 - time1)




