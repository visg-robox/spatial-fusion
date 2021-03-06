"""
Written by Zhang Jian

Test two situations:
1. window size = 1-50
2. time step is not continuous
"""
import sys
import common
sys.path.append(common.project_path)

import torch

import random
from model.rnn import *

from evaluate.eval_ssnet import *
from visualization.input_data import visualize_batch

from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


# Hyper Parameters
EPOCH = 100                             # train the training data n times, to save time, we just train 1 epoch
# when batch size = 1, we just want to have a test
BATCH_SIZE = 512  # common.batch_size
TIME_STEP = 50  # common.time_step                          # rnn time step / image height
INPUT_SIZE = common.feature_num         # rnn input size / image width
HIDDEN_SIZE = common.feature_num
OUTPUT_SIZE = common.class_num
LR = 0.001                              # learning rate
WINDOW_SIZE = 50

USING_RNN_FEATURE = common.USING_RNN_FEATURE
USING_SSNet_FEATURE = common.USING_SSNet_FEATURE


if __name__ == '__main__':

    data_path = common.data_path
    infer_path = data_path + 'CARLA_episode_0019/test2/infer_feature/'
    gt_path = data_path + 'CARLA_episode_0019/test2/gt_feature/'
    test_infer_path = data_path + 'CARLA_episode_0019/test2/test_feature/infer/'
    test_gt_path = data_path + 'CARLA_episode_0019/test2/test_feature/gt/'
    res_save_path = str(os.getcwd()) + '/runs/standard_feature'

    infer_file = get_file_list(infer_path)
    infer_file.sort()
    gt_file = get_file_list(gt_path)
    gt_file.sort()

    writer = SummaryWriter('runs/standard_feature')
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
                input_data = data_loader_torch.featuremap_to_batch(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                input_data = Variable(input_data, requires_grad=True).cuda()
                gt = data_loader_torch.featuremap_to_gt_num(gt_dict, current_keys, BATCH_SIZE)
                gt = Variable(gt).cuda()

                output = rnn(input_data, TIME_STEP)
                loss = loss_func(output, gt)
                optimizer.zero_grad()
                loss.backward()
                # for name, param in rnn.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), record_iter)
                optimizer.step()
                record_iter += 1
                writer.add_scalar('data/feature_training_loss', loss, record_iter)
                print(record_iter)
                if record_iter % 1000 == 0:
                    model_name = res_save_path + str(record_iter) + '_model.pkl'
                    torch.save(rnn, model_name)
                    eval_ssnet(test_infer_path, test_gt_path, model_name, res_save_path, WINDOW_SIZE, time_step=TIME_STEP)
    writer.close()

