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
from model import spnet
from data_process import data_balance

import time as timer

# Hyper Parameters

EPOCH = common.epoch                           # train the training data n times, to save time, we just train 1 epoch
# when batch size = 1, we just want to have a test
BATCH_SIZE = common.batch_size  # common.batch_size
TIME_STEP = common.time_step  # common.time_step                          # rnn time step / image height

INPUT_SIZE = common.feature_num_ivo         # rnn input size / image width
HIDDEN_SIZE = common.feature_num_ivo
OUTPUT_SIZE = common.class_num
NEAR_NUM = common.near_num

LR = common.lr                             # learning rate

Pretrained = common.pretrained

dataset_name = common.dataset_name
method_name = common.method_name


if __name__ == '__main__':

    train_path = os.path.join(common.blockfile_path, 'train')
    res_save_path = os.path.join(common.res_save_path, dataset_name, method_name)
    common.make_path(res_save_path)

    infer_file_list = common.get_file_list_with_pattern('infer_feature', train_path)
    gt_file_list = common.get_file_list_with_pattern('gt', train_path)
    if len(infer_file_list) == len(gt_file_list):
        file_len = len(infer_file_list)
    else:
        raise RuntimeError('infer_file number is not equal to gt_file number')

    writer = SummaryWriter(os.path.join(res_save_path, 'event'))
    if Pretrained is False:
        model = spnet.SPNet(INPUT_SIZE, INPUT_SIZE, OUTPUT_SIZE)
    else:
        pretrain_model_path = common.pre_train_model_path
        model = torch.load(pretrain_model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    random.seed(10)
    model.cuda()
    model.train()
    if Pretrained is False:
        record_iter = 0
    else:
        record_iter = int(common.pre_train_step)
    label_p = np.loadtxt(common.class_preserve_proba_path)
    for epoch in range(EPOCH):
        scheduler.step()
        print('Epoch: ', epoch)
        infer_index = np.arange(file_len)
        random.shuffle(infer_index)
        for time in range(file_len//common.file_num_step):
            file_idx_list = infer_index[time*common.file_num_step : (time+1) * common.file_num_step]
            voxel_dict = dict()
            gt_dict = dict()
            print('start reading file')
            readFileStartTime = timer.time()
            for file_idx in file_idx_list:
                infer_filename = infer_file_list[file_idx]
                gt_filename = gt_file_list[file_idx]
                if infer_filename.split('/')[-1] == gt_filename.split('/')[-1]:
                    print(infer_filename)
                else:
                    raise RuntimeError('infer_file and gt_file is different')
                voxel_dict.update(np.load(infer_filename, allow_pickle=True).item())
                gt_dict.update(np.load(gt_filename, allow_pickle=True).item())
            voxel_dict_res, gt_dict_res, keys_list = data_balance.data_balance_new(voxel_dict, gt_dict, label_p)
            # keys_list = common.get_common_keys(voxel_dict_res, gt_dict_res)
            print('finish reading file')
            readFileEndTime = timer.time()
            print('read file use: ')
            print(readFileEndTime-readFileStartTime)
            random.shuffle(keys_list)

            trainStartTime = timer.time()
            for i in range(len(keys_list)//BATCH_SIZE):
                current_keys = keys_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                input_data = data_loader_torch.featuremap_to_batch_ivo_with_neighbour(voxel_dict_res, current_keys, BATCH_SIZE, NEAR_NUM, TIME_STEP, INPUT_SIZE)
                input_data = Variable(input_data, requires_grad=True).cuda()
                gt = data_loader_torch.featuremap_to_gt_num(gt_dict, current_keys, BATCH_SIZE, common.ignore_list)
                gt = Variable(gt).cuda()

                output = model.forward(input_data)
                loss = loss_func(output, gt)
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                # for name, param in rnn.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), record_iter)
                optimizer.step()
                record_iter += 1
                if epoch % 10 ==0:
                    writer.add_scalar('data/feature_training_loss', loss, record_iter)
                print(record_iter)
                if record_iter % common.model_save_step == 0:
                    model_name = os.path.join(res_save_path, str(record_iter) + '_model.pkl')
                    torch.save(model, model_name)
                    #test_loss = eval_spnet_balance(test_infer_path, test_gt_path, model, res_save_path, WINDOW_SIZE, time_step=TIME_STEP, log_dir=res_save_path)
                    #writer.add_scalar('data/feature_test_loss', test_loss, record_iter)
            trainEndTime = timer.time()
            print('train use: ')
            print(trainEndTime-trainStartTime)


    writer.close()

