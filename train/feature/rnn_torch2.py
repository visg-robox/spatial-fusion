"""
Written by Zhang Jian

Test two situations:
1. window size = 1-50
2. time step is not continuous
"""
import sys
sys.path.append("/home/zhangjian/code/project/spatial-fusion/")
import common
import torch
from os.path import join
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
BATCH_SIZE = 64  # common.batch_size
TIME_STEP = common.time_step  # common.time_step                          # rnn time step / image height
INPUT_SIZE = common.feature_num_raw           # rnn input size / image width
HIDDEN_SIZE = common.feature_num_raw
OUTPUT_SIZE = common.class_num
LR = 0.001                              # learning rate
WINDOW_SIZE = 50
Pretrained = False
file_num = 3
model_path = None
USING_RNN_FEATURE = common.USING_RNN_FEATURE
USING_SSNet_FEATURE = common.USING_SSNet_FEATURE


if __name__ == '__main__':

    data_path = common.blockfile_path
    infer_path = join(data_path, 'infer_feature/')
    gt_path = join(data_path, 'gt_feature/')
    test_infer_path = join(data_path, 'test_feature/infer/')
    test_gt_path = join(data_path, 'test_feature/gt/')
    res_save_path = str(os.getcwd()) + '/runs/LSTM_imgfeature/'
    if not os.path.isdir(res_save_path):
        os.makedirs(res_save_path)

    infer_file = get_file_list(infer_path)
    infer_file.sort()
    gt_file = get_file_list(gt_path)
    gt_file.sort()

    writer = SummaryWriter(res_save_path)
    if Pretrained == False:
        rnn = SSNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    else:
        rnn = torch.load(model_path)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    random.seed(10)
    rnn.cuda()
    record_iter = 0
    label_p = np.loadtxt('../../data_process/data_dropout_ratio.txt')
    for epoch in range(EPOCH):
        scheduler.step()
        print('Epoch: ', epoch)
        file_idx_list_all = np.arange(len(infer_file))
        random.shuffle(file_idx_list_all)
        for time in range(len(infer_file)//file_num):
            file_idx_list = file_idx_list_all[time * file_num: (time + 1) * file_num]
            voxel_dict = dict()
            gt_dict = dict()
            print('start reading file')
            for file_idx in file_idx_list:
                infer_filename = infer_file[file_idx]
                gt_filename = gt_file[file_idx]
                voxel_dict.update(np.load(infer_filename).item())
                gt_dict.update(np.load(gt_filename).item())
            voxel_dict, gt_dict = data_balance.data_balance_rnn_new(voxel_dict, gt_dict, label_p)
            keys_list = get_common_keys(voxel_dict, gt_dict)
            print('finish reading file')

            random.shuffle(keys_list)
            for i in range(len(keys_list)//BATCH_SIZE):
                current_keys = keys_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                input_data = data_loader_torch.featuremap_to_batch_new(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
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
                if record_iter % 2500 == 0:
                    model_name = res_save_path + str(record_iter) + '_model.pkl'
                    torch.save(rnn, model_name)
                    #eval_ssnet(test_infer_path, test_gt_path, model_name, res_save_path, WINDOW_SIZE, time_step=TIME_STEP, log_dir=res_save_path)
    writer.close()

