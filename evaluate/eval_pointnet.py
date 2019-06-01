

from data_process import data_loader_torch
from data_process.data_process_feature import *
from evaluate.eval_API import *

import torch
from data_process import data_balance
from torch import nn
from torch.autograd import Variable


def eval_pointnet(
               test_gt_path,
               model_path,
               log_dir='.'):

    test_gt_file_list = get_file_list(test_gt_path)
    test_gt_file_list.sort()

    loss_func = nn.CrossEntropyLoss()
    model = torch.load(model_path)
    model.cuda()
    model.eval()


    total_accuracy_rnn = np.zeros([common.class_num, 3], dtype=np.float32)

    test_pred_y = np.zeros(1, dtype=int)
    test_gt_y = np.array([-255], dtype=int)
    for test_file_idx in range(50):

        test_gt_filename = test_gt_file_list[test_file_idx]
        block_res, gt_res = data_loader_torch.pointnet_block_process_xyzlocal(test_gt_filename)
        input_data = Variable(torch.FloatTensor(block_res), requires_grad=False).cuda()
        input_data = input_data.permute(0, 2, 1)
        test_gt = Variable(torch.LongTensor(gt_res)).cuda()
        with torch.no_grad():
            test_output, _, _ = model(input_data)
        test_pred_y = numpy.append(test_pred_y, torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze())
        test_gt_y = numpy.append(test_gt_y, test_gt.cpu().numpy())
        accuracy_rnn = getaccuracy(test_pred_y, test_gt_y, common.class_num)
        total_accuracy_rnn += accuracy_rnn

    evaluate_name = 'pointnet'
    eval_print_save(total_accuracy_rnn, evaluate_name, log_dir)


if __name__ == '__main__':
    data_path = common.blockfile_path
    test_infer_path = os.path.join(data_path, 'test', 'infer_feature')
    test_gt_path = test_infer_path.replace('infer_feature', 'gt')
    model_path = '/media/luo/Dataset/RnnFusion/spatial-fusion/train/feature/result/apollo_record001/pointnet_feature_tranform_batch_size16/1200_model.pkl'
    save_dir = '/media/luo/Dataset/RnnFusion/spatial-fusion/train/feature/result/apollo_record001/pointnet_feature_tranform_batch_size16'
    eval_pointnet(test_gt_path, model_path, log_dir = save_dir)