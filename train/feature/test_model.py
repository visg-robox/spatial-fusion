import torch
import common
import random
from model.rnn import *
from data_structure.voxel_map import *
from data_process import data_loader_torch
from data_process.data_process_feature import *
from evaluate.eval_API import *
from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    work_root_path = '/home/zhangjian/code/project/RnnFusion/'
    model_path = work_root_path + 'record/feature/20k-lr0.01-fix-linear/model.pkl'
    result_path = 'record/feature/20k-lr0.01-fix-linear/'
    rnn = torch.load(model_path)
    train_pred_y = np.zeros(1, dtype=int)
    train_gt_y = np.zeros(1, dtype=int)

    for file_num in range(len(infer_file)):
        infer_filename = infer_file[file_num]
        gt_filename = gt_file[file_num]
        voxel_dict = np.load(infer_filename).item()
        gt_dict = np.load(gt_filename).item()
        keys_list = list(voxel_dict.keys())
        print('infer file name: ', infer_filename)
        for i in range(len(keys_list) // BATCH_SIZE):
            start_time = time.time()
            current_keys = keys_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            input_data = data_loader_torch.featuremap_to_batch(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP,
                                                               INPUT_SIZE)
            input_data = Variable(input_data, requires_grad=True).cuda()
            gt = data_loader_torch.featuremap_to_gt_num(gt_dict, current_keys, BATCH_SIZE)
            gt = Variable(gt).cuda()

            output = rnn(input_data, TIME_STEP)
            train_pred_y = numpy.append(train_pred_y, torch.max(output.cpu(), 1)[1].data.numpy().squeeze())
            train_gt_y = numpy.append(train_gt_y, gt.cpu().numpy())
    total_accuracy_rnn = getaccuracy(train_pred_y, train_gt_y, common.class_num)
    eval_print_save(total_accuracy_rnn, 'training_miou_result_condition_rnn', '.')
