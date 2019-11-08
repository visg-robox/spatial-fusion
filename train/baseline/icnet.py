import sys
sys.path.append("../../")
from data_process.data_loader_torch import *
from model_evaluate.eval_API import *
from random import choice


def icnet(semantic_map, batch_size, key_list):
    res = numpy.zeros(batch_size)
    for i in range(len(key_list)):
        key = key_list[i]
        semantic_info = semantic_map.find_key(key).semantic_info_list[0]
        res[i] = int(numpy.argmax(semantic_info.label_list))
    return res


def make_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


dataset_name = common.dataset_name
method_name = common.method_name


if __name__ == '__main__':
    test_path = os.path.join(common.blockfile_path, 'test')
    res_save_path = os.path.join(common.res_save_path, dataset_name, method_name)
    make_path(res_save_path)

    infer_file_list = common.get_file_list_with_pattern('infer_label', test_path)
    gt_file_list = common.get_file_list_with_pattern('gt', test_path)

    infer_res = []
    gt_res = []

    if len(infer_file_list) == len(gt_file_list):
        file_len = len(infer_file_list)
    else:
        raise RuntimeError('infer_file number is not equal to gt_file number')

    # compare random choice from gt_voxel and infer voxel
    for i in range(file_len):
        infer_name = infer_file_list[i]
        gt_name = gt_file_list[i]
        if infer_name.split('/')[-1] == gt_name.split('/')[-1]:
            print(infer_name)
        else:
            raise RuntimeError('infer_file and gt_file is different')
        infer_map = np.load(infer_name).item()
        gt_map = np.load(gt_name).item()
        infer_keys_list = list(infer_map.keys())
        gt_keys_list = list(gt_map.keys())
        keys_list = [v for v in infer_keys_list if v in gt_keys_list]
        for key in keys_list:
            infer_voxel = infer_map[key]
            gt_voxel = gt_map[key]
            gt_label = choice(gt_voxel.feature_info_list).feature_list[0]
            if gt_label in common.ignore_list:
                gt_label = int(-100)
            gt_res.append(int(gt_label))
            infer_label = choice(infer_voxel.feature_info_list).feature_list
            #infer_label = choice(infer_voxel.semantic_info_list).label_list
            # infer_res.append(infer_label.argmax())
            infer_res.append(np.array(infer_label).argmax())
    total_accuracy = getaccuracy(infer_res, gt_res, common.class_num)
    eval_print_save(total_accuracy, common.method_name, res_save_path)
