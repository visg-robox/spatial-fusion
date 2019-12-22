"""
Written by Zhang Jian

Test two situations:
1. time step = 1
2. time step is not continuous
"""

import sys
import time
import multiprocessing

sys.path.append("../")

from data_process import data_loader_torch
from data_process.data_process_feature import *
from model_evaluate.eval_API import *

import torch
from data_process import data_balance
from torch import nn
from torch.autograd import Variable
from functools import partial
import xlrd

# Hyper Parameters
TEST_BATCH_SIZE = common.test_batch_size
TIME_STEP = common.time_step  # rnn time step / image height
if "lstm_iv" in common.method_name:
    INPUT_SIZE = common.feature_num_iv  # rnn input size / image width
    HIDDEN_SIZE = common.feature_num_iv
elif "lstm_i" in common.method_name:
    INPUT_SIZE = common.feature_num_i  # rnn input size / image width
    HIDDEN_SIZE = common.feature_num_i
else:
    INPUT_SIZE = common.feature_num_ivo  # rnn input size / image width
    HIDDEN_SIZE = common.feature_num_ivo


def eval_spnet_balance(test_path,
                       model_path,
                       time_step=TIME_STEP,
                       log_dir='.',
                       ignore_list=[]):
    test_infer_file_list = common.get_file_list_with_pattern('infer_feature', test_path)
    test_gt_file_list = common.get_file_list_with_pattern('gt', test_path)
    if len(test_infer_file_list) == len(test_gt_file_list):
        file_len = len(test_infer_file_list)
    else:
        raise RuntimeError('infer_file number is not equal to gt_file number')

    loss_func = nn.CrossEntropyLoss()
    rnn = torch.load(model_path)
    rnn.cuda()
    rnn.eval()
    test_pred_y = np.zeros(1, dtype=int)
    test_gt_y = np.array([255], dtype=int)
    # 不能使用1,必须使用255使初始是一个无效的voxel

    test_loss_all = 0
    # for test_file_idx in range(5):
    for test_file_idx in range(file_len):
        test_infer_filename = test_infer_file_list[test_file_idx]
        test_gt_filename = test_gt_file_list[test_file_idx]
        test_infer_dict = np.load(test_infer_filename, allow_pickle=True).item()
        test_gt_dict = np.load(test_gt_filename, allow_pickle=True).item()
        label_p = np.ones(common.class_num)
        if "lstm" not in common.method_name:
            test_infer_dict, test_gt_dict = data_balance.data_balance(test_infer_dict, test_gt_dict, label_p)
        test_keys_list = common.get_common_keys(test_infer_dict, test_gt_dict)
        print('test file: ', test_infer_filename)
        test_loss_ave = 0
        time1 = time.time()
        for j in range(len(test_keys_list) // TEST_BATCH_SIZE):
            test_current_keys = test_keys_list[j * TEST_BATCH_SIZE:(j + 1) * TEST_BATCH_SIZE]
            if "lstm_iv" in common.method_name:
                test_input = data_loader_torch.featuremap_to_batch_iv(test_infer_dict,
                                                                      test_current_keys,
                                                                      TEST_BATCH_SIZE,
                                                                      time_step,
                                                                      INPUT_SIZE)
                test_input = Variable(test_input).cuda()
                test_output = rnn(test_input, time_step)

            elif "lstm_i" in common.method_name:
                test_input = data_loader_torch.featuremap_to_batch_i(test_infer_dict,
                                                                     test_current_keys,
                                                                     TEST_BATCH_SIZE,
                                                                     time_step,
                                                                     INPUT_SIZE)
                test_input = Variable(test_input).cuda()
                test_output = rnn(test_input, time_step)

            else:
                test_input = data_loader_torch.featuremap_to_batch_ivo_with_neighbour(test_infer_dict,
                                                                                      test_current_keys,
                                                                                      TEST_BATCH_SIZE,
                                                                                      common.near_num,
                                                                                      time_step,
                                                                                      INPUT_SIZE)
                test_input = Variable(test_input).cuda()
                test_output = rnn(test_input)
            with torch.no_grad():
                test_input = test_input
            test_gt = data_loader_torch.featuremap_to_gt_num(test_gt_dict,
                                                             test_current_keys,
                                                             TEST_BATCH_SIZE,
                                                             ignore_list=ignore_list)

            # test_loss = loss_func(test_output, test_gt.cuda())
            # test_loss_ave += test_loss
            test_pred_y = numpy.append(test_pred_y, torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze())
            test_gt_y = numpy.append(test_gt_y, test_gt.cpu().numpy())
        time2 = time.time()
        print(time2 - time1)

    total_accuracy_rnn = getaccuracy(test_pred_y, test_gt_y, common.class_num)
    if sys.argv[2] == 'test':
        evaluate_name = model_path.split('/')[-1].split('.')[0]
        eval_print_save(total_accuracy_rnn, evaluate_name, log_dir)
        return test_loss_all
    else:
        per_class_accuracy = total_accuracy_rnn[:, 1] / total_accuracy_rnn[:, 2]
        mean_accuracy = np.sum(total_accuracy_rnn[:, 1]) / np.sum(total_accuracy_rnn[:, 2])
        per_class_iou = total_accuracy_rnn[:, 1] / (
        total_accuracy_rnn[:, 0] + total_accuracy_rnn[:, 2] - total_accuracy_rnn[:, 1])
        index = np.where(np.greater(total_accuracy_rnn[:, 2], 0))
        new_iou = per_class_iou[index]
        miou = np.mean(new_iou)
        return mean_accuracy, miou


def eval_spnet_balance_multi_process(all_path,
                                     model, model_path,
                                     time_step=TIME_STEP,
                                     log_dir='.',
                                     ignore_list=[]):
    test_infer_file = all_path[0]
    test_gt_file = all_path[1]

    '''
    if len(test_infer_file_list) == len(test_gt_file_list):
        file_len = len(test_infer_file_list)
    else:
        raise RuntimeError('infer_file number is not equal to gt_file number')
    '''

    loss_func = nn.CrossEntropyLoss()

    test_pred_y = np.zeros(1, dtype=int)
    test_gt_y = np.array([255], dtype=int)
    # 不能使用1,必须使用255使初始是一个无效的voxel

    test_loss_all = 0
    # for test_file_idx in range(5):

    test_infer_filename = test_infer_file
    test_gt_filename = test_gt_file
    test_infer_dict = np.load(test_infer_filename, allow_pickle=True).item()
    test_gt_dict = np.load(test_gt_filename, allow_pickle=True).item()
    label_p = np.ones(common.class_num)
    if "lstm" not in common.method_name:
        test_infer_dict, test_gt_dict = data_balance.data_balance(test_infer_dict, test_gt_dict, label_p)
    test_keys_list = common.get_common_keys(test_infer_dict, test_gt_dict)
    print('test file: ', test_infer_filename)
    test_loss_ave = 0
    time_b = time.time()
    for j in range(len(test_keys_list) // TEST_BATCH_SIZE):
        test_current_keys = test_keys_list[j * TEST_BATCH_SIZE:(j + 1) * TEST_BATCH_SIZE]
        if "lstm_iv" in common.method_name:
            test_input = data_loader_torch.featuremap_to_batch_iv(test_infer_dict,
                                                                  test_current_keys,
                                                                  TEST_BATCH_SIZE,
                                                                  time_step,
                                                                  INPUT_SIZE)
            test_input = Variable(test_input).cuda()
            test_output = model(test_input, time_step)

        elif "lstm_i" in common.method_name:
            test_input = data_loader_torch.featuremap_to_batch_i(test_infer_dict,
                                                                 test_current_keys,
                                                                 TEST_BATCH_SIZE,
                                                                 time_step,
                                                                 INPUT_SIZE)
            test_input = Variable(test_input).cuda()
            test_output = model(test_input, time_step)

        else:
            test_input = data_loader_torch.featuremap_to_batch_ivo_with_neighbour(test_infer_dict,
                                                                                  test_current_keys,
                                                                                  TEST_BATCH_SIZE,
                                                                                  common.near_num,
                                                                                  time_step,
                                                                                  INPUT_SIZE)
            test_input = Variable(test_input).cuda()
            test_output = model(test_input)
        with torch.no_grad():
            test_input = test_input
        test_gt = data_loader_torch.featuremap_to_gt_num(test_gt_dict,
                                                         test_current_keys,
                                                         TEST_BATCH_SIZE,
                                                         ignore_list=ignore_list)

        # test_loss = loss_func(test_output, test_gt.cuda())
        # test_loss_ave += test_loss
        test_pred_y = numpy.append(test_pred_y, torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze())
        test_gt_y = numpy.append(test_gt_y, test_gt.cpu().numpy())
    time_e = time.time()
    print(time_e - time_b)

    total_accuracy_rnn = getaccuracy(test_pred_y, test_gt_y, common.class_num)

    evaluate_name = model_path.split('/')[-1].split('.')[0] + test_infer_filename.split('/')[-1].split('.')[0]

    eval_print_save(total_accuracy_rnn, evaluate_name, log_dir,
                    points_num=(len(test_keys_list) // TEST_BATCH_SIZE) * TEST_BATCH_SIZE)
    return test_loss_all


def eval_spnet(model_path):
    test_path = os.path.join(common.blockfile_path, 'test')
    print(model_path)
    save_path = os.path.dirname(model_path)
    loss = eval_spnet_balance(test_path, model_path, time_step=common.time_step, log_dir=save_path,
                              ignore_list=common.ignore_list)


def do(all_path, model, model_path, save_path):
    loss = eval_spnet_balance_multi_process(all_path, model, model_path, time_step=common.time_step, log_dir=save_path,
                                            ignore_list=common.ignore_list)
    return loss


def eval_spnet_multi_process(model_path, scene_name="Record006/"):
    time1 = time.time()
    print(model_path)
    save_path = os.path.join(os.path.dirname(model_path), 'eval_result')
    print(save_path)
    common.make_path(save_path)
    data_path = os.path.join(common.blockfile_path, 'test', scene_name, 'infer_feature')
    gt_path = os.path.join(common.blockfile_path, 'test', scene_name, 'gt')

    data_source_path = common.get_file_list(data_path)
    data_source_path.sort()
    gt_source_path = common.get_file_list(gt_path)
    gt_source_path.sort()

    length = len(data_source_path)

    all_paths = []
    for i in range(length):
        all_paths.append([data_source_path[i], gt_source_path[i]])

    model = torch.load(model_path)
    model.cuda()
    model.eval()

    pool = multiprocessing.Pool(processes=2)
    partial_do = partial(do, model=model, model_path=model_path, save_path=save_path)
    pool.map(partial_do, all_paths)

    pool.close()
    pool.join()
    time2 = time.time()
    print("Sub-process(es) done.")
    print(time2 - time1)

    excel_file_list = os.listdir(save_path)
    class_dict = {}  # {class_name: class_acc, class_iou, class_percentage}
    num_points_all = 0
    for i in range(len(excel_file_list)):
        points_num = int(excel_file_list[i].split('_')[-2])
        excel_file_name = os.path.join(save_path, excel_file_list[i])

        wb = xlrd.open_workbook(excel_file_name)
        sheet1 = wb.sheet_by_index(0)
        class_name = sheet1.row_values(0)
        class_acc = sheet1.row_values(1)
        class_iou = sheet1.row_values(4)
        class_percentage = sheet1.row_values(7)

        for index in range(len(class_name) - 3):
            real_index = index + 3
            if class_name[real_index] in class_dict:
                [past_acc, past_iou, past_percentage] = class_dict[class_name[real_index]]
                new_percentage = (past_percentage * num_points_all + class_percentage * points_num) / (
                num_points_all + points_num)
                new_acc = (
                          past_percentage * num_points_all * past_acc + class_percentage * points_num * class_acc) / new_percentage * (
                          num_points_all + points_num)
                new_iou = 0
                class_dict[class_name[real_index]] = [new_acc, new_iou, new_percentage]
            else:
                class_dict[class_name[real_index]] = [class_acc, class_iou, class_percentage]
        num_points_all += points_num


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_relative_path', type = str)

    FLAGS, unparsed = parser.parse_known_args()

    test_path = os.path.join(common.blockfile_path, 'test')

    model_path = os.path.join(common.test_model_path, FLAGS.model_relative_path)

    print(model_path)
    save_path = os.path.dirname(model_path)
    # import sys
    # sys.path.append("/media/zhangjian/U/RnnFusion")
    # eval_ssnet(test_infer_path, test_gt_path, model_path, res_path, window_size=20, time_step=20)
    # eval_ssnet_cell(test_infer_path, test_gt_path, model_path, input_window=5, time_step=20)
    loss = eval_spnet_balance(test_path, model_path, time_step=common.time_step, log_dir=save_path, ignore_list = common.ignore_list)

'''
