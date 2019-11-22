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
EPOCH = common.epoch                          # train the training data n times, to save time, we just train 1 epoch
# when batch size = 1, we just want to have a test
BATCH_SIZE = common.batch_size  # common.batch_size
TIME_STEP = common.time_step  # common.time_step                          # rnn time step / image height
INPUT_SIZE = common.feature_num_iv          # rnn input size / image width
HIDDEN_SIZE = common.feature_num_iv
OUTPUT_SIZE = common.class_num
LR = common.lr                              # learning rate
FILE_NUM_STEP = common.file_num_step


dataset_name = common.dataset_name
method_name = common.method_name
Pretrained = common.pretrained

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

    if Pretrained is False:
        rnn = SSNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    else:
        pretrain_model_path = common.pre_train_model_path
        print(pretrain_model_path)
        rnn = torch.load(pretrain_model_path)

    writer = SummaryWriter(os.path.join(res_save_path, 'event'))
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    random.seed(10)
    rnn.cuda()
    if Pretrained is False:
        record_iter = 0
    else:
        record_iter = int(common.pre_train_step)
    label_p = np.loadtxt(common.class_preserve_proba_path)
    for epoch in range(EPOCH):
        scheduler.step()
        print('Epoch: ', epoch)
        for time in range(file_len//common.file_num_step):
            file_idx_list = random.sample(range(file_len), common.file_num_step)
            voxel_dict = dict()
            gt_dict = dict()
            print('start reading file')
            for file_idx in file_idx_list:
                infer_filename = infer_file_list[file_idx]
                gt_filename = gt_file_list[file_idx]
                if infer_filename.split('/')[-1] == gt_filename.split('/')[-1]:
                    print(infer_filename)
                else:
                    raise RuntimeError('infer_file and gt_file is different')
                voxel_dict.update(np.load(infer_filename, allow_pickle=True).item())
                gt_dict.update(np.load(gt_filename, allow_pickle=True).item())
            keys_list = data_balance.data_balance_rnn(voxel_dict, gt_dict, label_p)
            print('finish reading file')
            random.shuffle(keys_list)
            for i in range(len(keys_list)//BATCH_SIZE):
                current_keys = keys_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
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
                total_norm=nn.utils.clip_grad_norm(rnn.parameters(),2)
                optimizer.step()
                record_iter += 1
                writer.add_scalar('data/feature_training_total_norm', total_norm, record_iter)
                writer.add_scalar('data/feature_training_loss', loss, record_iter)
                print(record_iter)
                if record_iter % common.model_save_step == 0:
                    model_name = os.path.join(res_save_path, str(record_iter) + '_model.pkl')
                    torch.save(rnn, model_name)
                    #train_mean_accuracy,train_miou=eval_spnet_balance(train_path,model_name)
                    #writer.add_scalar('data/train_mean_accuracy', train_mean_accuracy, record_iter)
                    #writer.add_scalar('data/train_miou', train_miou, record_iter)
                #    eval_ssnet(test_infer_path, test_gt_path, model_name, res_save_path, WINDOW_SIZE, time_step=TIME_STEP, log_dir=res_save_path)
    writer.close()

