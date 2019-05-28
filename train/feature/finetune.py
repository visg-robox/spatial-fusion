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

# Hyper Parameters
EPOCH = 100                             # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = common.batch_size
TIME_STEP = common.time_step                          # rnn time step / image height
INPUT_SIZE = common.feature_num_ivo         # rnn input size / image width
LR = 0.001                              # learning rate


USING_RNN_FEATURE = common.USING_RNN_FEATURE
USING_SSNet_FEATURE = common.USING_SSNet_FEATURE
TRAIN_ACCURACY = False
# TRAINING = True
# TESTING = False


if __name__ == '__main__':

    data_path = common.data_path
    infer_path = data_path + 'CARLA_episode_0019/test2/infer_feature/'
    gt_path = data_path + 'CARLA_episode_0019/test2/gt_feature/'

    infer_file = get_file_list(infer_path)
    infer_file.sort()
    gt_file = get_file_list(gt_path)
    gt_file.sort()

    test_inferPath = data_path + 'CARLA_episode_0019/test2/test_feature/infer/'
    test_gtPath = data_path + 'CARLA_episode_0019/test2/test_feature/gt/'

    test_inferFile = get_file_list(test_inferPath)
    test_inferFile.sort()
    test_GtFile = get_file_list(test_gtPath)
    test_GtFile.sort()

    # infer_file = test_inferFile
    # gt_file = test_GtFile

    writer = SummaryWriter('runs/finetune')

    if USING_SSNet_FEATURE is True:
        rnn = SSNet(INPUT_SIZE, common.feature_num_ivo, common.class_num)
    if USING_RNN_FEATURE is True:
        rnn = Rnn(INPUT_SIZE, TIME_STEP, 2)

    rnn.free_linear()
    rnn.cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    model_path = data_path + 'train/feature/59000_model.pkl'
    pretrain_model = torch.load(model_path)
    pretrain_dict = pretrain_model.state_dict()
    rnn_dict = rnn.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in rnn_dict}
    
    rnn_dict.update(pretrain_dict)
    rnn.load_state_dict(pretrain_dict)

    record_iter = 0
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        scheduler.step()
        for num in range(len(infer_file)//5):
            file_idx_list = random.sample(range(len(infer_file)), 10)
            voxel_dict = dict()
            gt_dict = dict()
            for file_idx in file_idx_list:
                infer_filename = infer_file[file_idx]
                gt_filename = gt_file[file_idx]
                voxel_dict.update(np.load(infer_filename).item())
                gt_dict.update(np.load(gt_filename).item())
            infer_keys_list = list(voxel_dict.keys())
            gt_keys_list = list(gt_dict.keys())
            keys_list = [v for v in infer_keys_list if v in gt_keys_list]
            print('infer file name: ', infer_filename[0])
            for i in range(len(keys_list)//BATCH_SIZE):
                start_time = time.time()
                # current_keys = keys_list[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
                current_keys = random.sample(keys_list, BATCH_SIZE)
                input_data = data_loader_torch.featuremap_to_batch(voxel_dict, current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                input_data = Variable(input_data, requires_grad=True).cuda()
                gt = data_loader_torch.featuremap_to_gt_num(gt_dict, current_keys, BATCH_SIZE)
                gt = Variable(gt).cuda()

                if USING_RNN_FEATURE is True:
                    output = rnn(input_data)
                if USING_SSNet_FEATURE is True:
                    output = rnn(input_data, TIME_STEP)
                loss = loss_func(output, gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                end_time = time.time()
                used_time = end_time - start_time
                writer.add_scalar('data/feature_training_loss', loss, record_iter)
                record_iter += 1
                print(record_iter)

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
                                data_loader_torch.featuremap_to_batch(test_voxel_dict, test_current_keys, BATCH_SIZE, TIME_STEP, INPUT_SIZE)
                            test_input_data = Variable(test_input_data).cuda()
                            test_gt = data_loader_torch.featuremap_to_gt_num(test_gt_dict, test_current_keys, BATCH_SIZE)
                            test_gt = Variable(test_gt).cuda()
                            if USING_SSNet_FEATURE is True:
                                test_output = rnn(test_input_data, TIME_STEP)
                            if USING_RNN_FEATURE is True:
                                test_output = rnn(test_input_data)
                            test_loss = loss_func(test_output, test_gt)
                            writer.add_scalar('data/feature_test_loss', test_loss, record_iter/1000)
                            # test_loss = loss_func(test_output, test_gt.cuda())
                            test_pred_y = numpy.append(test_pred_y, torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze())
                            test_gt_y = numpy.append(test_gt_y, test_gt.cpu().numpy())
                    model_name = str(record_iter) + '_model.pkl'
                    torch.save(rnn, model_name)
                    accuracy = float((test_pred_y == test_gt_y).astype(int).sum()) / float(test_gt_y.size)
                    total_accuracy_rnn = getaccuracy(test_pred_y, test_gt_y, common.class_num)
                    evaluate_name = str(record_iter) + '_test_miou_result_condition_rnn_feature'
                    eval_print_save(total_accuracy_rnn, evaluate_name, '.')
    writer.close()

