import os, shutil
import common
import numpy as np

def get_file_list(data_dir):
    path_list = list()
    for i in os.listdir(data_dir):
        path_list.append(os.path.join(data_dir, i))
    return path_list


def make_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)

def to_distance(name, initial_pose):
    tmp = np.zeros(initial_pose.shape)
    for i in range(len(name.split('_'))):
        tmp[i] = int(name.split('_')[i])
    return np.linalg.norm(tmp - initial_pose)

if __name__ == "__main__":
    data_path = common.data_path
    infer_path = data_path + 'CARLA_episode_0019/test3/infer_feature/'
    gt_path = data_path + 'CARLA_episode_0019/test3/gt_feature/'
    test_infer_path = data_path + 'CARLA_episode_0019/test3/test_feature/infer/'
    test_gt_path = data_path + 'CARLA_episode_0019/test3/test_feature/gt/'
    make_path(test_infer_path)
    make_path(test_gt_path)

    #initial_pose = np.loadtxt('./raw.txt')
    initial_pose = np.zeros([3])
    infer_file = get_file_list(infer_path)
    infer_file.sort(key=lambda x:(to_distance(x.split('/')[-1].split('.')[0], initial_pose)))
    gt_file = get_file_list(gt_path)
    gt_file.sort(key=lambda x:(to_distance(x.split('/')[-1].split('.')[0], initial_pose)))

    move_length = len(infer_file)//5

    infer_file_move = infer_file[-move_length:]
    gt_file_move = gt_file[-move_length:]
    for item in infer_file_move:
        cur_infer_file = item
        fpath, fname = os.path.split(cur_infer_file)
        shutil.move(cur_infer_file, test_infer_path + fname)

    for item in gt_file_move:
        cur_gt_file = item
        fpath, fname = os.path.split(cur_gt_file)
        shutil.move(cur_gt_file, test_gt_path + fname)
