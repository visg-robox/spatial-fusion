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

from evaluate.eval_ssnet import *
from visualization.input_data import visualize_batch

from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from model.pointnet_v2 import PointNetDenseCls
from data_process import data_balance, data_loader_torch
import math
import time
# Hyper Parameters

EPOCH = 100
SAVE_STEP = 100 # train the training data n times, to save time, we just train 1 epoch
# when batch size = 1, we just want to have a test
BATCH_SIZE = 16  # common.batch_size
Pretrained = common.pretrained
dataset_name = common.dataset_name
LR = 1e-2
method_name = 'pointnet_feature_tranform_batch_size16_newbalance'
Sample_num = 10000


def make_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


if __name__ == '__main__':

    data_path = common.blockfile_path
    infer_path = os.path.join(data_path, 'infer_feature')
    gt_path = os.path.join(data_path, 'gt')
    # test_infer_path = data_path + 'CARLA_episode_0019/test3/test_feature/infer/'
    # test_gt_path = data_path + 'CARLA_episode_0019/test3/test_feature/gt/'
    res_save_path = os.path.join(common.res_save_path, dataset_name, method_name)
    make_path(res_save_path)

    pretrain_model_path = res_save_path

    label_p = np.loadtxt(common.class_preserve_proba_path)
    weight = np.zeros_like(label_p)
    for i in range(label_p.size):
        if label_p[i] > 0:
            weight[i] = label_p[i] / (np.sum(label_p) /np.sum(np.greater(label_p, 0))) * 10


    print(weight)
    infer_file = get_file_list(infer_path)
    infer_file.sort()
    gt_file = get_file_list(gt_path)
    gt_file.sort()

    writer = SummaryWriter(os.path.join(res_save_path,'event'))
    if Pretrained == False:
        model =PointNetDenseCls(k = common.class_num, feature_transform= True)
    else:
        model = torch.load(pretrain_model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    loss_func = nn.CrossEntropyLoss(weight = torch.Tensor(weight).cuda())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    random.seed(10)
    model.cuda()
    model.train()
    record_iter = 0
    #label_p = np.loadtxt(common.class_preserve_proba_path)
    for epoch in range(EPOCH):
        scheduler.step()
        print('Epoch: ', epoch)
        file_num_step = common.file_num_step
        infer_index = np.arange(len(infer_file))
        random.shuffle(infer_index)
        for i in range(len(infer_file)//BATCH_SIZE):
            file_idx_list = infer_index[i*BATCH_SIZE : (i+1) * BATCH_SIZE]
            print('start reading file')
            batch_block = []
            gt_block = []
            time1 = time.time()
            for file_idx in file_idx_list:
                gt_filename = gt_file[file_idx]
                block_res, gt_res = data_loader_torch.pointnet_block_process_xyzlocal(gt_filename, Sample_num)
                batch_block.append(block_res)
                gt_block.append(gt_res)
            batch_block = np.concatenate(batch_block, axis = 0)
            batch_gt = np.concatenate(gt_block, axis = 0)
            input_data = Variable(torch.FloatTensor(batch_block),  requires_grad=True).cuda()
            gt = Variable(torch.LongTensor(batch_gt)).cuda()
            print('finish reading')
            time2 = time.time()
            print(time2 - time1)
            input_data = input_data.permute(0,2,1)
            output,_,_ = model.forward(input_data)
            loss = loss_func(output, gt)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            # for name, param in rnn.named_parameters():
            #     writer.add_histogram(name, param.clone().cpu().data.numpy(), record_iter)
            optimizer.step()
            record_iter += 1
            if epoch % 10 ==0:
                writer.add_scalar('data/loss', loss, record_iter)
            print(record_iter)
            if record_iter % SAVE_STEP == 0:
                model_name = os.path.join(res_save_path, str(record_iter) + '_model.pkl')
                torch.save(model, model_name)
                #test_loss = eval_spnet_balance(test_infer_path, test_gt_path, model, res_save_path, WINDOW_SIZE, time_step=TIME_STEP, log_dir=res_save_path)
                #writer.add_scalar('data/feature_test_loss', test_loss, record_iter)

    writer.close()

