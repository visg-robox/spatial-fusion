from data_process.data_loader_torch import *
from evaluate.eval_API import *
from random import choice


def icnet(semantic_map, batch_size, key_list):
    res = numpy.zeros(batch_size)
    for i in range(len(key_list)):
        key = key_list[i]
        semantic_info = semantic_map.find_key(key).semantic_info_list[0]
        res[i] = int(numpy.argmax(semantic_info.label_list))
    return res


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

    # compare random choice from gt_voxel and infer voxel
    for i in range(len(infer_path_list)):
        infer_name = infer_path_list[i]
        gt_name = gt_path_list[i]
        print(infer_name)
        infer_map = np.load(infer_name).item()
        gt_map = np.load(gt_name).item()
        infer_keys_list = list(infer_map.keys())
        gt_keys_list = list(gt_map.keys())
        keys_list = [v for v in infer_keys_list if v in gt_keys_list]
        for key in keys_list:
            infer_voxel = infer_map[key]
            gt_voxel = gt_map[key]
            gt_label = choice(gt_voxel.semantic_info_list).label_list[0]
            gt_res.append(int(gt_label))
            infer_label = choice(infer_voxel.semantic_info_list).label_list
            infer_res.append(infer_label.argmax())
    total_accuracy = getaccuracy(infer_res, gt_res, common.class_num)
    eval_print_save(total_accuracy, 'miou_result_icnet', '.')
