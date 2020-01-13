# import common
from torch import nn
import torch
from random import choice
from model.rnn import *
from torch.autograd import Variable
from data_process.data_process_label import *
from data_process import data_loader_torch,data_balance
import sys
sys.path.append('../spatial-fusion/')

BATCH_SIZE = 32
#这里用来修改可视化的中心坐标以及范围
#Range = np.array([common.region_x, common.region_y, common.region_z]) * common.block_len


# input: data file path, result file path, fusion method
# output: write .txt file
def visualize_pc(data_path, gt_path = None, save_path = '.', fusion_method = 0, model_path=None):
    if model_path:
        # rnn = SSNet(common.feature_num, common.feature_num, common.class_num)
        rnn = torch.load(model_path)
        rnn.cuda()
        rnn.eval()
    map_name = common.FsMethod(fusion_method).name + '_res'
    data_source_path = common.get_file_list(data_path)
    data_source_path.sort()
    gt_source_path = common.get_file_list(gt_path)
    gt_source_path.sort()

    for num,source_filename in enumerate(data_source_path):
        position = source_filename.split('/')[-1].split('.')[0].split('_')
        position = np.array(position,dtype=np.int)

        visionlize_center = [position[i]/100 for i in range(len(position))]
        result_map = VoxelMap(visionlize_center)

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
                    output = rnn(input_data,common.time_step)
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

        if fusion_method is common.FsMethod.GT:
            result_map.write_map(save_path, map_name+str(num), onehot = False)
        else:
            result_map.write_map(save_path, map_name+str(num), onehot=True)


# True, False
VISUALIZE_STF = True
VISUALIZE_RNN_FEATURE = False
VISUALIZE_RNN_LABEL   = False
VISUALIZE_BAYES       = False
VISUALIZE_BASELINE    = False
VISUALIZE_GT = False

scene_name = 'Record006/'
if __name__ == '__main__':
    time1 = time.time()
    if VISUALIZE_STF:

        #model_path = '../train/feature/runs/SPNET/100000/100000newdata_model.pkl'
        model_path = common.test_model_path + '/spnet_res_100000_model.pkl'
        save_path = os.path.join(common.test_model_path, 'visual_spnet')
        make_dir(save_path)
        data_path  = os.path.join(common.blockfile_path, 'test/' +scene_name +'infer_feature/')
        gt_path = os.path.join(common.blockfile_path, 'test/' +scene_name +'gt/')
        visualize_pc(data_path, gt_path, save_path, common.FsMethod.STF, model_path)

    if VISUALIZE_RNN_FEATURE:
        root_path = common.blockfile_path
        model_path = common.test_model_path + '/lstm_15000_model.pkl'
        save_path = os.path.join(common.test_model_path, 'visual_lstm')
        make_dir(save_path)
        data_path = os.path.join(common.blockfile_path, 'test/' +scene_name +'infer_feature/')
        gt_path = os.path.join(common.blockfile_path, 'test/' +scene_name +'gt/')
        visualize_pc(data_path, gt_path, save_path, common.FsMethod.RNN_FEATURE, model_path)

    if VISUALIZE_BAYES:
        save_path = os.path.join(common.test_model_path, 'visual_bayes')
        make_dir(save_path)
        data_path = os.path.join(common.blockfile_path, 'test/' +scene_name +'infer_label/')
        gt_path = os.path.join(common.blockfile_path, 'test/' + scene_name + 'gt/')
        visualize_pc(data_path, gt_path, save_path, common.FsMethod.BAYES)


    if VISUALIZE_BASELINE:
        data_path = os.path.join(common.blockfile_path, 'test/' +scene_name +'infer_feature/')
        gt_path = os.path.join(common.blockfile_path, 'test/' +scene_name +'gt/')
        save_path = os.path.join(common.test_model_path, 'visual_deeplab')
        make_dir(save_path)
        visualize_pc(data_path, gt_path, save_path, common.FsMethod.BASELINE)

    if VISUALIZE_GT:
        save_path = os.path.join(common.test_model_path, 'visual')
        make_dir(save_path)
        data_path = os.path.join(common.blockfile_path, 'test/' +scene_name +'infer_feature/')
        visualize_pc(data_path, data_path, save_path, common.FsMethod.GT)

    time2 = time.time()
    print(time2 - time1)