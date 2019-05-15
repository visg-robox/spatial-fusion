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

    for i in range(len(infer_file)):
        if i % 5 is 0:
            cur_infer_file = infer_file[i]
            cur_gt_file = gt_file[i]
            fpath, fname = os.path.split(cur_infer_file)
            shutil.move(cur_infer_file, test_infer_path + fname)
            shutil.move(cur_gt_file, test_gt_path + fname)

