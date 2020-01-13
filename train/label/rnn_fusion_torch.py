import sys
sys.path.append("/home/zhangjian/code/project/spatial-fusion")


from data_process import data_loader_torch
from model.rnn import *
import numpy as np
from train.baseline.icnet import *
from tensorboardX import SummaryWriter


# Hyper Parameters
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = common.batch_size
TIME_STEP = common.time_step         # rnn time step / image height
INPUT_SIZE = 13         # rnn input size / image width
LR = 0.001               # learning rate

USING_SSNet = True
USING_RNN = False
TRAIN_ACCURACY = False

if __name__ == '__main__':
    data_path = '/home/zhangjian/code/project/data/CARLA_episode_0019/test2/'
    infer_path = data_path + 'infer/'
    gt_path = data_path + 'gt/'

    infer_file = get_file_list(infer_path)
    infer_file.sort()
    gt_file = get_file_list(gt_path)
    gt_file.sort()

    test_inferPath = data_path + 'test1/infer/'
    test_gtPath = data_path + 'test1/gt/'

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

    if USING_SSNet is True:
        rnn = SSNet(INPUT_SIZE, TIME_STEP, common.class_num)
    if USING_RNN is True:
        rnn = Rnn(INPUT_SIZE, TIME_STEP, 2)
    rnn.cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    record_iter = 0
    for epoch in range(EPOCH):
        for file_num in range(len(infer_file)):
            infer_filename = infer_file[file_num]
            gt_filename = gt_file[file_num]
            voxel_dict = np.load(infer_filename).item()
            gt_dict = np.load(gt_filename).item()
            keys_list = list(voxel_dict.keys())
            print('infer file name: ', infer_filename)
            if TRAIN_ACCURACY and (record_iter % 1000 == 0):
                train_pred_y = np.zeros(1, dtype=int)
                train_gt_y = np.zeros(1, dtype=int)
            for i in range(len(keys_list)//BATCH_SIZE):
                start_time = time.time()
                current_keys = keys_list[i * BATCH_SIZE:(i+1) * BATCH_SIZE]

                input_data = data_loader_torch.labelmap_to_batch(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                input_data = Variable(input_data, requires_grad=True).cuda()
                gt = data_loader_torch.labelmap_to_gt_num(gt_dict, current_keys, BATCH_SIZE)
                gt = Variable(gt).cuda()

                if USING_RNN is True:
                    output = rnn(input_data)
                if USING_SSNet is True:
                    output = rnn(input_data, TIME_STEP)

                loss = loss_func(output, gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                end_time = time.time()
                used_time = end_time - start_time

                writer.add_scalar('data/loss', loss, record_iter)
                record_iter += 1
                print(record_iter)
                if TRAIN_ACCURACY:
                    train_pred_y = numpy.append(train_pred_y, torch.max(output.cpu(), 1)[1].data.numpy().squeeze())
                    train_gt_y = numpy.append(train_gt_y, gt.cpu().numpy())
                if (record_iter % 200 == 0) and TRAIN_ACCURACY:
                    accuracy = float((train_pred_y == train_gt_y).astype(int).sum()) / float(train_gt_y.size)
                    total_accuracy_rnn = getaccuracy(train_pred_y, train_gt_y, common.class_num)
                    eval_print_save(total_accuracy_rnn, 'training_miou_result_condition_rnn', '.')

                if record_iter % 1000 == 0:
                    test_pred_y = np.zeros(1, dtype=int)
                    test_gt_y = np.zeros(1, dtype=int)
                    for test_file_num in range(len(test_inferFile)):
                        test_infer_filename = test_inferFile[test_file_num]
                        test_gt_filename = test_GtFile[test_file_num]
                        test_voxel_dict = np.load(test_infer_filename).item()
                        test_gt_dict = np.load(test_gt_filename).item()
                        test_keys_list = list(test_voxel_dict.keys())
                        print('test file: ', test_infer_filename)
                        for j in range(len(test_keys_list) // BATCH_SIZE):
                            test_current_keys = test_keys_list[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                            test_input_data = \
                                data_loader_torch.labelmap_to_batch(test_voxel_dict, test_current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                            test_input_data = Variable(test_input_data).cuda()
                            test_gt = data_loader_torch.labelmap_to_gt_num(test_gt_dict, test_current_keys, BATCH_SIZE)
                            if USING_SSNet is True:
                                test_output = rnn(test_input_data, TIME_STEP)
                            if USING_RNN is True:
                                test_output = rnn(test_input_data)
                            # test_loss = loss_func(test_output, test_gt.cuda())
                            test_pred_y = numpy.append(test_pred_y, torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze())
                            test_gt_y = numpy.append(test_gt_y, test_gt.numpy())

                    torch.save(rnn, 'model.pkl')
                    accuracy = float((test_pred_y == test_gt_y).astype(int).sum()) / float(test_gt_y.size)
                    total_accuracy_rnn = getaccuracy(test_pred_y, test_gt_y, common.class_num)
                    eval_print_save(total_accuracy_rnn, 'test_miou_result_condition_rnn', '.')
                    # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

            # if i % 100 == 0:
            #     prob_pred_y = ProbabilityFusion.bayesian_fusion(test_infer, test_data_size, test_keys, INPUT_SIZE)
            #     #prob_pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            #     accuracy = float((prob_pred_y.numpy() == test_gt.numpy()).astype(int).sum()) / float(test_gt.numpy().size)
            #     total_accuracy_pro = getaccuracy(prob_pred_y.numpy(), test_gt.numpy(), common.class_num)
            #     eval_print_save(total_accuracy_pro, 'miou_result_bayesian', '.')
            #     print('bayesian accuracy: %.4f' % accuracy)
            #
            # if i% 100 == 0:
            #     icnet_pred_y = icnet(test_infer, test_data_size, test_keys)
            #     accuracy = float((icnet_pred_y == test_gt.numpy()).astype(int).sum()) / float(test_gt.numpy().size)
            #     total_accuracy_icnet = getaccuracy(icnet_pred_y, test_gt.numpy(), common.class_num)
            #     eval_print_save(total_accuracy_icnet, 'miou_result_icnet', '.')
            #     print('icnet accuracy: %.4f' % accuracy)
