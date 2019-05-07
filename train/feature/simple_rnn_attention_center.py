"""
Written by Zhang Jian

Test two situations:
1. window size = 1-50
2. time step is not continuous

This realization has some mistakes, it will be removed
"""
import sys
sys.path.append("/home/zhangjian/code/project/RnnFusion")

import torch
import common
import random
from model.rnn_attention import *

from evaluate.eval_ssnet import *
from visualization.input_data import visualize_batch

from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


# Hyper Parameters
EPOCH = 100                             # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1  # common.batch_size
TIME_STEP = 50  # common.time_step                          # rnn time step / image height
INPUT_SIZE = common.feature_num_dist         # rnn input size / image width
HIDDEN_SIZE = common.feature_num_dist
OUTPUT_SIZE = common.class_num
LR = 0.001                              # learning rate
WINDOW_SIZE = 50

USING_RNN_FEATURE = common.USING_RNN_FEATURE
USING_SSNet_FEATURE = common.USING_SSNet_FEATURE


if __name__ == '__main__':

    root_path = '/home/wangkai/project2/RnnFusion/'
    inferPath = root_path + 'data/CARLA_episode_0019/test2/infer_feature/'
    gtPath = root_path + 'data/CARLA_episode_0019/test2/gt_feature/'
    test_infer_path = root_path + 'data/CARLA_episode_0019/test2/test_feature/infer/'
    test_gt_path = root_path + 'data/CARLA_episode_0019/test2/test_feature/gt/'
    res_save_path = str(os.getcwd()) + '/'

    infer_file = get_file_list(inferPath)
    infer_file.sort()
    gt_file = get_file_list(gtPath)
    gt_file.sort()

    writer = SummaryWriter('runs/feature')
    rnn = SSNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    random.seed(10)
    rnn.cuda()
    record_iter = 0
    for epoch in range(EPOCH):
        scheduler.step()
        print('Epoch: ', epoch)
        for time in range(len(infer_file)//1):
            file_idx_list = random.sample(range(len(infer_file)), 1)
            voxel_dict = dict()
            gt_dict = dict()
            print('start reading file')
            for file_idx in file_idx_list:
                infer_filename = infer_file[file_idx]
                gt_filename = gt_file[file_idx]
                voxel_dict.update(np.load(infer_filename).item())
                gt_dict.update(np.load(gt_filename).item())
            keys_list = get_common_keys(voxel_dict, gt_dict)
            print('finish reading file')

            for i in range(len(keys_list)//BATCH_SIZE):
                current_keys = random.sample(keys_list, BATCH_SIZE)
                input_data = data_loader_torch.featuremap_to_batch_with_dist(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                gt = data_loader_torch.featuremap_to_gt_num(gt_dict, current_keys, BATCH_SIZE)
                gt = Variable(gt).cuda()

                # network forward and backward
                time_step = input_data.size(1)
                loss_list = []
                #for window_step in range(time_step - WINDOW_SIZE + 1):
                # for count in range(20):
                cur_input = Variable(input_data, requires_grad=True).cuda()
                    # cur_input = Variable(input_data, requires_grad=True).cuda()
                    # if np.sum(cur_input.cpu().detach()[0, :, 0].numpy()) < 5:
                    #     continue
                    # visualize_batch(cur_input.cpu().detach()[0, :, :])
                    # output1 = torch.empty(1, OUTPUT_SIZE)
                    # output2 = torch.empty(1, OUTPUT_SIZE)
                output = rnn(cur_input, WINDOW_SIZE)
                    # output1[0, :] = output[0]
                    # output2[0, :] = output[1]
                    # gt1 = torch.empty(1, dtype=torch.long)
                    # gt2 = torch.empty(1, dtype=torch.long)
                    # gt1[0] = gt[0]
                    # gt2[0] = gt[1]
                loss = loss_func(output, gt)
                loss_list.append(loss)
                optimizer.zero_grad()
                loss.backward()
                for name, param in rnn.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), record_iter)
                optimizer.step()
                record_iter += 1
                writer.add_scalar('data/feature_training_loss', loss, record_iter)
                # writer.add_scalar('data/feature_training_loss', loss_list[-1], record_iter)
                # record_iter += 1
                print(record_iter)
                if record_iter % 1000 == 0:
                    model_name = res_save_path + str(record_iter) + '_model.pkl'
                    torch.save(rnn, model_name)
                    eval_ssnet(test_infer_path, test_gt_path, model_name, res_save_path, WINDOW_SIZE, time_step=TIME_STEP)
    writer.close()

