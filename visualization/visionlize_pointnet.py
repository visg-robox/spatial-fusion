from random import choice
from model.rnn import *
from torch.autograd import Variable
from data_process.data_process_label import *
from data_process import data_loader_torch,data_balance
import sys
from model_evaluate.eval_semantic_apollo import ID_COLOR
sys.path.append('/media/luo/Dataset/RnnFusion/spatial-fusion/')


def visionlize_pointnet(test_gt_path,  model_path, sample_num,):
    test_gt_file_list = get_file_list(test_gt_path)
    test_gt_file_list.sort()
    test_gt_file_list = test_gt_file_list[0:sample_num]

    model = torch.load(model_path)
    model.cuda()
    model.eval()
    
    for num, test_gt_filename  in enumerate(test_gt_file_list):
        block_res, gt_res, point_cloud = data_loader_torch.pointnet_block_process_xyzlocal_z(test_gt_filename)
        input_data_o = Variable(torch.FloatTensor(block_res), requires_grad=False).cuda()
        input_data = input_data_o.permute(0, 2, 1)
        test_gt = Variable(torch.LongTensor(gt_res)).cuda()
        with torch.no_grad():
            test_output, _, _ = model(input_data)
        test_pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy().squeeze()
        test_gt_y =  test_gt.cpu().numpy()
        
        model_dir = os.path.dirname(model_path)
        save_infer_dir = os.path.join(model_dir, 'visionlize/infer')
        save_gt_dir = os.path.join(model_dir, 'visionlize/gt')
        if not os.path.isdir(save_infer_dir):
            os.makedirs(save_infer_dir)
        if not os.path.isdir(save_gt_dir):
            os.makedirs(save_gt_dir)
        
        with open(os.path.join(save_infer_dir, str(num) + '.txt'), 'w') as infer_visionlize:
            with open(os.path.join(save_gt_dir, str(num) + '.txt'),
                      'w') as gt_visionlize:
                for i in range(point_cloud.shape[0]):
                     color_i = ID_COLOR[test_pred_y[i]]
                     color_g = ID_COLOR[test_gt_y[i]]
                     point = point_cloud[i]
                     line_infer = str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ' + \
                           str(color_i[0]) + ' ' + str(color_i[1]) + ' ' + str(color_i[2]) + '\n'
                     line_gt = str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ' + \
                           str(color_g[0]) + ' ' + str(color_g[1]) + ' ' + str(color_g[2]) + '\n'
                     infer_visionlize.write(line_infer)
                     gt_visionlize.write(line_gt)
                     
if __name__ == '__main__':
    test_gt_path = '/media/luo/Dataset/RnnFusion/apollo_data/processed_data/test/gt'
    model_path = '/media/luo/Dataset/RnnFusion/spatial-fusion/train/feature/result/apollo_record001/pointnet_feature_tranform_batch_size16/1300_model.pkl'
    visionlize_pointnet(test_gt_path, model_path, 200)