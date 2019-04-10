from data_process.data_loader_torch import *
from evaluate.eval_API import *
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


if __name__ == '__main__':
    rootPath = '/home/zhangjian/code/data/CARLA_episode_0019/test2/test1/'
    inferPath = rootPath + 'infer/'
    gtPath = rootPath + 'gt/'

    infer_path_list = get_file_list(inferPath)
    infer_path_list.sort()
    gt_path_list = get_file_list(gtPath)
    gt_path_list.sort()

    infer_res = []
    gt_res = []

    for i in range(len(infer_path_list)):
        infer_name = infer_path_list[i]
        gt_name = gt_path_list[i]
        print(infer_name)
        infer_map = np.load(infer_name).item()
        gt_map = np.load(gt_name).item()
        gt_keys = gt_map.keys()
        for key in gt_keys:
            infer_voxel = infer_map[key]
            gt_voxel = gt_map[key]
            gt_res.append(int(gt_voxel.semantic_info_list[0].label_list[0]))
            label_fusion = [1 for _ in range(common.class_num)]
            for idx in range(len(infer_voxel.semantic_info_list)):
                label_fusion = [a * b for a, b in zip(label_fusion, infer_voxel.semantic_info_list[idx].label_list)]
            infer_res.append(int(numpy.argmax(label_fusion)))
    total_accuracy = getaccuracy(infer_res, gt_res, common.class_num)
    eval_print_save(total_accuracy, 'miou_result_bayesian', '.')
