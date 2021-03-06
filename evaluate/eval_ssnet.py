"""
Written by Zhang Jian

Test two situations:
1. time step = 1
2. time step is not continuous
"""

import sys
sys.path.append("/home/zhangjian/code/project/RnnFusion")

from data_process import data_loader_torch
from data_process.data_process_feature import *
from evaluate.eval_API import *

import torch
from torch.autograd import Variable


# Hyper Parameters
TEST_BATCH_SIZE = 50
TIME_STEP = common.time_step                          # rnn time step / image height
INPUT_SIZE = common.feature_num         # rnn input size / image width
HIDDEN_SIZE = common.feature_num


def get_common_keys(infer_dict, gt_dict):
    infer_keys_list = list(infer_dict.keys())
    gt_keys_list = list(gt_dict.keys())
    common_keys_list = [v for v in infer_keys_list if v in gt_keys_list]
    return common_keys_list


# 这个测试的时候是存在问题的，最后一次的输入可能全都是0
def eval_ssnet(test_infer_path,
               test_gt_path,
               model_path,
               res_path,
               window_size,
               time_step=TIME_STEP):
    test_infer_file_list = get_file_list(test_infer_path)
    test_infer_file_list.sort()
    test_gt_file_list = get_file_list(test_gt_path)
    test_gt_file_list.sort()

    rnn = torch.load(model_path)
    rnn.cuda()

    test_pred_y = np.zeros(1, dtype=int)
    test_gt_y = np.zeros(1, dtype=int)

    # for test_file_idx in range(3):
    for test_file_idx in range(len(test_infer_file_list)):
        test_infer_filename = test_infer_file_list[test_file_idx]
        test_gt_filename = test_gt_file_list[test_file_idx]
        test_infer_dict = np.load(test_infer_filename).item()
        test_gt_dict = np.load(test_gt_filename).item()
        test_keys_list = get_common_keys(test_infer_dict, test_gt_dict)
        print('test file: ', test_infer_filename)
        for j in range(len(test_keys_list) // TEST_BATCH_SIZE):
            test_current_keys = test_keys_list[j * TEST_BATCH_SIZE:(j + 1) * TEST_BATCH_SIZE]
            test_input_data = data_loader_torch.featuremap_to_batch(test_infer_dict,
                                                                    test_current_keys,
                                                                    TEST_BATCH_SIZE,
                                                                    time_step,
                                                                    INPUT_SIZE)
            test_gt = data_loader_torch.featuremap_to_gt_num(test_gt_dict,
                                                             test_current_keys,
                                                             TEST_BATCH_SIZE)

            test_gt = Variable(test_gt).cuda()
            test_time_step = test_input_data.size(1)
            test_output_list = []
            for window_step in range(test_time_step - window_size + 1):
                test_cur_input = Variable(test_input_data[:, window_step:window_step+window_size, :], requires_grad=True).cuda()
                test_output = rnn(test_cur_input, window_size)
                test_output_list.append(test_output)
            test_output = test_output_list[-1]
            # test_loss = loss_func(test_output, test_gt.cuda())
            test_pred_y = numpy.append(test_pred_y, torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze())
            test_gt_y = numpy.append(test_gt_y, test_gt.cpu().numpy())

    # model_path = res_path + 'window_size_' + str(window_size) + '_model.pkl'
    # torch.save(rnn, model_path)
    # accuracy = float((test_pred_y == test_gt_y).astype(int).sum()) / float(test_gt_y.size)
    total_accuracy_rnn = getaccuracy(test_pred_y, test_gt_y, common.class_num)
    evaluate_name = 'window_size_' + str(window_size) + '_current_rnn_feature'
    eval_print_save(total_accuracy_rnn, evaluate_name, '.')


# 这个测试的时候是存在问题的，最后一次的输入可能全都是0
def eval_ssnet_cell(test_infer_path,
                    test_gt_path,
                    model_path,
                    input_window,
                    time_step=TIME_STEP):
    test_infer_file_list = get_file_list(test_infer_path)
    test_infer_file_list.sort()
    test_gt_file_list = get_file_list(test_gt_path)
    test_gt_file_list.sort()

    rnn = torch.load(model_path)
    rnn.cuda()

    test_pred_y = np.zeros(1, dtype=int)
    test_gt_y = np.zeros(1, dtype=int)

    # for test_file_idx in range(3):
    for test_file_idx in range(len(test_infer_file_list)):
        test_infer_filename = test_infer_file_list[test_file_idx]
        test_gt_filename = test_gt_file_list[test_file_idx]
        test_infer_dict = np.load(test_infer_filename).item()
        test_gt_dict = np.load(test_gt_filename).item()
        test_keys_list = get_common_keys(test_infer_dict, test_gt_dict)
        print('test file: ', test_infer_filename)
        for j in range(len(test_keys_list) // TEST_BATCH_SIZE):
            test_current_keys = test_keys_list[j * TEST_BATCH_SIZE:(j + 1) * TEST_BATCH_SIZE]
            test_input_data = data_loader_torch.featuremap_to_batch(test_infer_dict,
                                                                    test_current_keys,
                                                                    TEST_BATCH_SIZE,
                                                                    time_step,
                                                                    INPUT_SIZE)
            test_gt = data_loader_torch.featuremap_to_gt_num(test_gt_dict,
                                                             test_current_keys,
                                                             TEST_BATCH_SIZE)

            test_gt = Variable(test_gt).cuda()
            test_time_step = test_input_data.size(1)
            test_output_list = []
            h0_t = Variable(torch.zeros(TEST_BATCH_SIZE, HIDDEN_SIZE), requires_grad=True).cuda()
            c0_t = Variable(torch.zeros(TEST_BATCH_SIZE, HIDDEN_SIZE), requires_grad=True).cuda()
            h1_t = Variable(torch.zeros(TEST_BATCH_SIZE, HIDDEN_SIZE), requires_grad=True).cuda()
            c1_t = Variable(torch.zeros(TEST_BATCH_SIZE, HIDDEN_SIZE), requires_grad=True).cuda()

            for window_step in range(test_time_step-input_window+1):
                test_cur_input = Variable(test_input_data[:, window_step:window_step+input_window, :], requires_grad=True).cuda()
                test_output, h0_t, c0_t, h1_t, c1_t = rnn(test_cur_input, h0_t, c0_t, h1_t, c1_t)
                test_output_list.append(test_output)
                h0_t = Variable(h0_t.data, requires_grad=True).cuda()
                c0_t = Variable(c0_t.data, requires_grad=True).cuda()
                h1_t = Variable(h1_t.data, requires_grad=True).cuda()
                c1_t = Variable(c1_t.data, requires_grad=True).cuda()
            test_output = test_output_list[-1]
            # test_loss = loss_func(test_output, test_gt.cuda())
            test_pred_y = numpy.append(test_pred_y, torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze())
            test_gt_y = numpy.append(test_gt_y, test_gt.cpu().numpy())

    # accuracy = float((test_pred_y == test_gt_y).astype(int).sum()) / float(test_gt_y.size)
    total_accuracy_rnn = getaccuracy(test_pred_y, test_gt_y, common.class_num)
    evaluate_name = 'current_rnn_2_feature'
    eval_print_save(total_accuracy_rnn, evaluate_name, '.')


if __name__ == '__main__':
    data_path = '/home/zhangjian/code/project/RnnFusion/'
    test_infer_path = data_path + 'data/CARLA_episode_0019/test2/test_feature/infer/'
    test_gt_path = data_path + 'data/CARLA_episode_0019/test2/test_feature/gt/'
    res_path = './model/'
    # model_path = data_path + 'train/feature/exe/window_size_50/window_size_50_iter_73000_model.pkl'
    model_path = '/home/zhangjian/code/project/RnnFusion/train/feature/180000_model.pkl'
    # import sys
    # sys.path.append("/media/zhangjian/U/RnnFusion")
    # eval_ssnet(test_infer_path, test_gt_path, model_path, res_path, window_size=20, time_step=20)
    eval_ssnet_cell(test_infer_path, test_gt_path, model_path, input_window=5, time_step=20)