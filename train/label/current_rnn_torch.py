import sys
sys.path.append("/home/zhangjian/code/project/spatial-fusion")

import random
from data_process import data_loader_torch
from model.rnn import *
import numpy as np
from train.baseline.icnet import *
from tensorboardX import SummaryWriter


# Hyper Parameters
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = common.batch_size
TIME_STEP = common.time_step         # rnn time step / image height
WINDOW_SIZE = 50
INPUT_SIZE = 13         # rnn input size / image width
HIDDEN_SIZE = 100
OUTPUT_SIZE = common.class_num
LR = 0.001               # learning rate

USING_SSNet = True
USING_RNN = False
TRAIN_ACCURACY = False

if __name__ == '__main__':

    data_path = '/home/zhangjian/code/project/data/CARLA_episode_0019/test2/'

    infer_path = data_path + 'infer/'
    gt_path = data_path + 'gt/'

    test_inferPath = data_path + 'test1/infer/'
    test_gtPath = data_path + 'test1/gt/'

    # infer_path = test_inferPath
    # gt_path = test_gtPath

    infer_file = get_file_list(infer_path)
    infer_file.sort()
    gt_file = get_file_list(gt_path)
    gt_file.sort()

    test_inferFile = get_file_list(test_inferPath)
    test_inferFile.sort()
    test_GtFile = get_file_list(test_gtPath)
    test_GtFile.sort()

    # tensorboard
    writer = SummaryWriter('runs/label')

    # test data
    # test data location
    test_data_size = 10000
    test_infer_path = data_path + 'test/infer/'
    test_gt_path = data_path + 'test/gt/'

    # # pre-process
    # test_keys = test_keys_list[0 * test_data_size:(0 + 1) * test_data_size]
    # test_input = data_loader_torch.labelmap_to_batch(test_infer, test_keys, test_data_size, TIME_STEP, INPUT_SIZE)
    # # last input
    # test_input = Variable(test_input, requires_grad=True)
    # test_gt = data_loader_torch.labelmap_to_gt_num(test_gt, test_keys, test_data_size)

    rnn = SSNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    rnn.cuda()
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=LR,  weight_decay=1e-5)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    record_iter = 0
    for epoch in range(EPOCH):
        scheduler.step()
        for file_num in range(len(infer_file)):
            file_idx_list = random.sample(range(len(infer_file)), 1)
            voxel_dict = dict()
            gt_dict = dict()
            print('start reading file')
            for file_idx in file_idx_list:
                infer_filename = infer_file[file_idx]
                gt_filename = gt_file[file_idx]
                voxel_dict.update(np.load(infer_filename).item())
                gt_dict.update(np.load(gt_filename).item())
            infer_keys_list = list(voxel_dict.keys())
            gt_keys_list = list(gt_dict.keys())
            keys_list = [v for v in infer_keys_list if v in gt_keys_list]
            print('finish reading file')
            for i in range(len(keys_list)//BATCH_SIZE):
                start_time = time.time()
                current_keys = random.sample(keys_list, BATCH_SIZE)
                input_data = data_loader_torch.labelmap_to_batch(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                # input_data = Variable(input_data, requires_grad=True).cuda()
                loss_list = []
                gt = data_loader_torch.labelmap_to_gt_num(gt_dict, current_keys, BATCH_SIZE)
                gt = Variable(gt).cuda()
                for j in range(TIME_STEP - WINDOW_SIZE + 1):
                    cur_input_data = Variable(input_data[:, j:j+WINDOW_SIZE, :], requires_grad=True).cuda()
                    # if int(cur_input_data[0, 0, 0].data[0].item()) is 0:
                    #     continue
                    output = rnn(cur_input_data, WINDOW_SIZE)
                    loss = loss_func(output, gt)
                    loss_list.append(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                used_time = end_time - start_time

                writer.add_scalar('data/train_loss', loss_list[-1], record_iter)
                record_iter += 1
                if record_iter % 10 == 0:
                    print(record_iter)

                if record_iter % 10000 == 0:
                    test_pred_y = np.zeros(1, dtype=int)
                    test_gt_y = np.zeros(1, dtype=int)
                    for test_file_num in range(len(test_inferFile)):
                        test_infer_filename = test_inferFile[test_file_num]
                        test_gt_filename = test_GtFile[test_file_num]
                        test_voxel_dict = np.load(test_infer_filename).item()
                        test_infer_keys_list = test_voxel_dict.keys()
                        test_gt_dict = np.load(test_gt_filename).item()
                        test_gt_keys_list = test_gt_dict.keys()
                        # test_keys_list = [v for v in test_infer_keys_list if v in test_gt_keys_list]
                        test_keys_list = list(test_voxel_dict.keys())
                        print('test file: ', test_infer_filename)
                        for j in range(len(test_keys_list) // BATCH_SIZE):
                            test_current_keys = test_keys_list[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                            test_input_data = \
                                data_loader_torch.labelmap_to_batch(test_voxel_dict, test_current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                            test_input_data = Variable(test_input_data).cuda()
                            # test_gt = data_loader_torch.labelmap_to_gt_num(test_gt_dict, test_current_keys, BATCH_SIZE)
                            test_output_list = []
                            test_gt = data_loader_torch.labelmap_to_gt_num(test_gt_dict, test_current_keys, BATCH_SIZE)

                            for l in range(TIME_STEP - WINDOW_SIZE + 1):
                                test_gt = Variable(test_gt).cuda()
                                test_cur_input_data = Variable(test_input_data[:, l:l+WINDOW_SIZE, :]).cuda()
                                # if int(test_cur_input_data[0, 0, 0].data[0].item()) is 0:
                                #     continue
                                test_output = rnn(test_cur_input_data, WINDOW_SIZE)
                                test_output_list.append(test_output)
                                # test_loss = loss_func(test_output, test_gt.cuda())
                            test_pred_y = numpy.append(test_pred_y, torch.max(test_output_list[-1].cpu(), 1)[1].data.numpy().squeeze())
                            test_gt_y = numpy.append(test_gt_y, test_gt.cpu().numpy())

                    model_name = str(record_iter) + '_model.pkl'
                    torch.save(rnn, model_name)
                    accuracy = float((test_pred_y == test_gt_y).astype(int).sum()) / float(test_gt_y.size)
                    total_accuracy_rnn = getaccuracy(test_pred_y, test_gt_y, common.class_num)
                    eval_print_save(total_accuracy_rnn, 'test_miou_result_condition_rnn', '.')
