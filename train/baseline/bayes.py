import sys
sys.path.append("../../")
from data_process.data_loader_torch import *
from model_evaluate.eval_API import *
from data_structure.voxel_map import *
from data_structure.voxel_semantic import *


class ProbabilityFusion:

    @staticmethod
    def bayesian_fusion(semantic_map, batch_size, key_list, input_size):
        res = torch.zeros(batch_size, dtype=torch.int64)
        for i in range(len(key_list)):
            key = key_list[i]
            semantic_info_list = semantic_map.find_key(key).semantic_info_list
            label_fusion = [1 for _ in range(input_size)]
            for j in range(len(semantic_info_list)):
                label_fusion = [a * b for a, b in zip(label_fusion, semantic_info_list[j].label_list)]
                label_fusion = softmax(label_fusion)
            res[i] = int(numpy.argmax(label_fusion))
        return res


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)
    x = numpy.array(x)
    x = numpy.exp(x)
    x.astype('float32')
    if x.ndim == 1:
        sumcol = sum(x)
        for i in range(x.size):
            x[i] = x[i] / float(sumcol)
    if x.ndim > 1:
        sumcol = x.sum(axis=0)
        for row in x:
            for i in range(row.size):
                row[i] = row[i] / float(sumcol[i])
    return x


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
    for i in range(file_len):
        infer_name = infer_file_list[i]
        gt_name = gt_file_list[i]
        if infer_name.split('/')[-1] == gt_name.split('/')[-1]:
            print(infer_name)
        else:
            raise RuntimeError('infer_file and gt_file is different')
        infer_map = np.load(infer_name, allow_pickle=True).item()
        gt_map = np.load(gt_name, allow_pickle=True).item()
        keys = common.get_common_keys(infer_map, gt_map)
        for key in keys:
            infer_voxel = infer_map[key]
            gt_voxel = gt_map[key]
            gt_res.append(int(gt_voxel.feature_info_list[0].feature_list[0]))
            label_fusion = [1 for _ in range(common.class_num)]
            try:
                for idx in range(len(infer_voxel.semantic_info_list)):
                    label_fusion = [a * b for a, b in zip(label_fusion, infer_voxel.semantic_info_list[idx].label_list)]
            except AttributeError:
                for idx in range(len(infer_voxel.feature_info_list)):
                    label_fusion = [a * b for a, b in zip(label_fusion, infer_voxel.feature_info_list[idx].feature_list)]
            infer_res.append(int(numpy.argmax(label_fusion)))
    total_accuracy = getaccuracy(infer_res, gt_res, common.class_num)
    eval_print_save(total_accuracy, common.method_name, res_save_path)