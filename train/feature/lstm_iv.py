"""
Written by Zhang Jian

Test two situations:
1. window size = 1-50
2. time step is not continuous
"""
import sys
sys.path.append("../../")
import common
import torch

import random
from model.rnn import *

from model_evaluate.eval_ssnet import *
from visualization.input_data import visualize_batch

from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


# Hyper Parameters
EPOCH = common.epoch                             # train the training data n times, to save time, we just train 1 epoch
# when batch size = 1, we just want to have a test
BATCH_SIZE = common.batch_size  # common.batch_size
TIME_STEP = common.time_step  # common.time_step                          # rnn time step / image height
INPUT_SIZE = common.feature_num_iv           # rnn input size / image width
HIDDEN_SIZE = common.feature_num_iv
OUTPUT_SIZE = common.class_num
LR = common.lr                            # learning rate
FILE_NUM_STEP = common.file_num_step



dataset_name = common.dataset_name
method_name = 'lstm_iv'


if __name__ == '__main__':

    data_path = common.blockfile_path
    infer_path = os.path.join(data_path, 'infer_feature')
    gt_path = os.path.join(data_path, 'gt')
    res_save_path = os.path.join(common.res_save_path, dataset_name, method_name)
    common.make_path(res_save_path)

    infer_file = get_file_list(infer_path)
    infer_file.sort()
    gt_file = get_file_list(gt_path)
    gt_file.sort()

    writer = SummaryWriter(os.path.join(res_save_path, 'event'))
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
        for time in range(len(infer_file)//FILE_NUM_STEP):
            file_idx_list = random.sample(range(len(infer_file)), FILE_NUM_STEP)
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
                input_data = data_loader_torch.featuremap_to_batch_iv(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                input_data = Variable(input_data, requires_grad=True).cuda()
                gt = data_loader_torch.featuremap_to_gt_num(gt_dict, current_keys, BATCH_SIZE, ignore_list=common.ignore_list)
                gt = Variable(gt).cuda()

                output = rnn(input_data, TIME_STEP)
                loss = loss_func(output, gt)
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                # for name, param in rnn.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), record_iter)
                optimizer.step()
                record_iter += 1
                writer.add_scalar('data/feature_training_loss', loss, record_iter)
                print(record_iter)
                if record_iter % common.model_save_step == 0:
                    model_name = os.path.join(res_save_path, str(record_iter) + '_model.pkl')
                    torch.save(rnn, model_name)
    writer.close()

