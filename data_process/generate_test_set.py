import os, shutil
import common



def get_file_list(data_dir):
    path_list = list()
    for i in os.listdir(data_dir):
        path_list.append(os.path.join(data_dir, i))
    return path_list


def make_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


if __name__ == "__main__":
    data_path = common.data_path
    infer_path = data_path + 'CARLA_episode_0019/test3/infer_feature/'
    gt_path = data_path + 'CARLA_episode_0019/test3/gt_feature/'
    test_infer_path = data_path + 'CARLA_episode_0019/test3/test_feature/infer/'
    test_gt_path = data_path + 'CARLA_episode_0019/test3/test_feature/gt/'
    make_path(test_infer_path)
    make_path(test_gt_path)

    infer_file = get_file_list(infer_path)
    infer_file.sort()
    gt_file = get_file_list(gt_path)
    gt_file.sort()
    count = 0
    count2 = 0
    for item in infer_file:
        count += 1
        if count % 5 == 0:
            cur_infer_file = item
            fpath, fname = os.path.split(cur_infer_file)
            shutil.move(cur_infer_file, test_infer_path + fname)

    for item in gt_file:
        count2 += 1
        if count2 % 5 == 0:
            cur_gt_file = item
            fpath, fname = os.path.split(cur_gt_file)
            shutil.move(cur_gt_file, test_gt_path + fname)

