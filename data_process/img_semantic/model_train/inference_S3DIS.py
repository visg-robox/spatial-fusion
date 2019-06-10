"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
from os.path import join
from os import listdir
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy import misc
import config, dataset_util
from utils import preprocessing
import matplotlib.pyplot as plt
from scipy import  misc
import sys
import json

sys.path.append('../../../')
import common
import glob
from data_process.generate_dataset import divide_multi_sequence
from dataset_script.S3DIS_scipt.assets.utils import *

# 需要修改的变量
_DATA_DIR = common.raw_data_path
_SAMPLE_NUM = common.point_num_per_frame
_ROOM_CLASS = ['WC', 'hallway', 'office', 'conferenceRoom', 'lobby', 'storage', 'pantry']
_ROOM_NUMBER = 1
_SAVE_DIR = common.blockfile_path


# 一般不用修改,输入图片的尺寸和相机５的内参
IMG_HEIGHT = 1080
IMG_WIDTH = 1080


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default=_DATA_DIR,
                    help='The directory containing the image data.')

parser.add_argument('--model_dir', type=str,
                    help='The directory containing the checkpoint')

parser.add_argument('--model', type=str, choices=['I', 'D'],
                    help='ICNET or Deeplab.')

parser.add_argument('--output_dir', type=str, default=_SAVE_DIR,
                    help='Path to the directory to generate the inference results')

parser.add_argument('--sample_point', type=str, default=_SAMPLE_NUM,
                    help='Path to the directory to generate the inference results')


_NUM_CLASSES = dataset_util.NUM_CLASSES

def read_examples_list(path):
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip() for line in lines]

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, list(range(N))
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample], sample
    else:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample]
        return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)


Coormat = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.float32)
for i in range(IMG_HEIGHT):
    for j in range(IMG_WIDTH):
        Coormat[i, j] = np.array([j, i, 1], dtype=np.float32)
Indexmap = Coormat.reshape([-1, 3])

    
def Image_map2pcl_global_S3DIS(rgb_path, sample_points):

    """
    通过rgb path 自动推断出其他所有path, 然后完成深度图到点云, 点云的sample以及
    :param rgb_path:
    :param sample_points:
    :return:
    """
    gt_sem_path, depth_path, pose_path = rgb_path2all_path_S3DIS(rgb_path)
    gt_semmap = decode_gt_S3DIS(gt_sem_path)
    depthmap = decode_depth_S3DIS(depth_path)
    extrincs, instrics = decode_extrincs_S3DIS(pose_path)
    depthmap = depthmap.reshape([-1])
    gt_semmap = gt_semmap.reshape([-1])
    #regular_index
    
    index = np.where(depthmap < 50)
    # 这里是
    index = np.array(list(index))
    index = np.squeeze(index)
    index, _ = sample_data(index, sample_points)
    
    depthmap = depthmap[index]
    indexmap = Indexmap[index]
    gt_semmap = gt_semmap[index]
    
    pc_local = np.dot(np.linalg.inv(instrics), np.transpose(indexmap)) * np.transpose(depthmap)
    pc_local = np.append(pc_local, np.ones((1, pc_local.shape[1])), axis=0)
    pc_global = np.dot(extrincs, pc_local)[0:3]
    pc_global = np.transpose(pc_global)
    return pc_global, gt_semmap, indexmap[:, :2], index, extrincs

def bilinear_interp_PointWithIndex(items, target_shape, xy):
    """

    :param items: 3 dim array
    :param xy:    xy_index
    :return:      points_value after bilinear interp
    
    """
    items_shape = items.shape[0:2]
    ratio = (np.array(items_shape, dtype=np.float32) - 1) / (np.array(target_shape, dtype=np.float32) - 1)
    ratio = np.array([ratio[1], ratio[0]], dtype=np.float32)
    xy = xy * ratio
    
    xy_floor = np.floor(xy)
    x_floor = xy_floor[:, 0]
    y_floor = xy_floor[:, 1]
    
    if (xy - xy_floor).any():
        xy_ceil = np.ceil(xy)
        x_ceil = xy_ceil[:, 0]
        y_ceil = xy_ceil[:, 1]
        # Get valid RGB index
        
        RGBindex_A = (np.int32(y_ceil), np.int32(x_floor))
        RGBindex_B = (np.int32(y_ceil), np.int32(x_ceil))
        RGBindex_C = (np.int32(y_floor), np.int32(x_ceil))
        RGBindex_D = (np.int32(y_floor), np.int32(x_floor))
        x0 = (xy[:, 0] - x_floor)
        x1 = (x_ceil - xy[:, 0])
        y0 = (xy[:, 1] - y_floor)
        y1 = (y_ceil - xy[:, 1])
        x0 = np.expand_dims(x0, axis=1)
        x1 = np.expand_dims(x1, axis=1)
        y0 = np.expand_dims(y0, axis=1)
        y1 = np.expand_dims(y1, axis=1)
        valueArray = items[RGBindex_A] * x1 * y0 + items[RGBindex_B] * x0 * y0 + items[
            RGBindex_C] * x0 * y1 + items[RGBindex_D] * x1 * y1
        
    else:
        index = (np.int32(y_floor), np.int32(x_floor))
        valueArray = items[index]
    return valueArray

def write_S3DIS_lidar_data(data_dir, phase, model):
    rgb_path_list_all = glob.glob(os.path.join(data_dir, phase, '*/data/rgb/*.png'))
    if phase == 'train':
        room_num = _ROOM_NUMBER * 5
    else:
        room_num = _ROOM_NUMBER
    for room_class in _ROOM_CLASS:
        rgb_path = [i for i in rgb_path_list_all if room_class in i]
        room_id_set = set(map(get_room_id, rgb_path))
        if len(room_id_set) > room_num:
            room_id_set = np.array(list(room_id_set))
            np.random.shuffle(room_id_set)
            room_id_set_sample = list(room_id_set[:room_num])
        else:
            room_id_set_sample = room_id_set
        for room_id in room_id_set_sample:
            rgb_list = [i for i in rgb_path if room_id in i]
            save_dir = os.path.join(_SAVE_DIR, phase, room_id)
            write_sequence_lidar_data(save_dir, rgb_list, model)

def write_sequence_lidar_data(sequcence_save_dir, RGB_list, model):
    pcl_feature_prefix = join(sequcence_save_dir, 'infer_feature')
    if not os.path.isdir(pcl_feature_prefix):
        os.makedirs(pcl_feature_prefix)
    pcl_p_prefix = join(sequcence_save_dir, 'infer_label')
    if not os.path.isdir(pcl_p_prefix):
        os.makedirs(pcl_p_prefix)
    pcl_gt_prefix = join(sequcence_save_dir, 'gt')
    if not os.path.isdir(pcl_gt_prefix):
        os.makedirs(pcl_gt_prefix)
    pcl_pose_prefix = join(sequcence_save_dir, 'pose')
    if not os.path.isdir(pcl_pose_prefix):
        os.makedirs(pcl_pose_prefix)
    pcl_prefix = join(sequcence_save_dir, 'lidar')
    if not os.path.isdir(pcl_prefix):
        os.makedirs(pcl_prefix)
    
    shutil.copy('inference_S3DIS.py', join(sequcence_save_dir, 'inference_S3DIS.py'))
    shutil.copy('train.py', join(sequcence_save_dir, 'train.py'))
    shutil.copy('config.py', join(sequcence_save_dir, 'config.py'))
    
    # 因为中间有的帧没有GT所以要先判断有没有GT，再加入列表中去
    List_path = sequcence_save_dir + '/list.txt'
    with open(List_path, 'w') as fp:
        for line in RGB_list:
            fp.write(line + '\n')
    # 一些支持的函数
    image_files = read_examples_list(List_path)
    predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=None)
    
    f = open(os.path.join(sequcence_save_dir, 'accuracy_record.txt'), 'w')
    for pred_dict, rgb_path in zip(predictions, image_files):
        # 路径规则
        frame = rgb_path.split('/')[-1].split('.')[0]
        # 获得采样之后的xy_index, 和铺平之后的index，以及采样完之后的点云
        pcl, gt_map, index_xy, index, extrincs= Image_map2pcl_global_S3DIS(rgb_path, FLAGS.sample_point)
        index_xy = np.array(index_xy, dtype=np.float32)
        
        # 输出和图片
        RGB = np.array(Image.open(rgb_path))
        pred_P = pred_dict['probabilities_origin']
        pred_ID = pred_dict['classes']
        pred_feature = pred_dict['feature_out']

        # 因为全分辨率的feature实在是太占显存了，所以feature返回不是全分辨率，要使用双线性插值来获得
        origin_shape = RGB.shape[0:2]
        point_feature = bilinear_interp_PointWithIndex(pred_feature, origin_shape, index_xy)
        point_probabilities = bilinear_interp_PointWithIndex(pred_P, origin_shape, index_xy)
        rgb = bilinear_interp_PointWithIndex(RGB, origin_shape, index_xy)
        
        #准确率用来debug
        debug_point_feature = np.argmax(point_probabilities, axis=1)
        total_num = np.sum(np.where(gt_map != 255))
        correct_num = np.sum(np.where(gt_map == debug_point_feature))
        accuracy = correct_num / total_num
        print(accuracy)
        gt_map = np.expand_dims(gt_map, axis=1)
        f.write(frame + '\t' + str(accuracy) + '\n')
        
        #需要保存的文件
        pred_ID = np.repeat(pred_ID, 3, axis=2)
        print(pred_ID.shape)
        pred_ID_PATH = join(sequcence_save_dir, 'Pictures/Infer/ID')
        if not os.path.isdir(pred_ID_PATH):
            os.makedirs(pred_ID_PATH)
        misc.imsave(join(pred_ID_PATH, frame + '.png'), np.uint8(pred_ID))
        #保存点云数据, 前三维是三维坐标
        np.savetxt(join(pcl_pose_prefix, frame), extrincs)
        pcl_feature = np.concatenate([pcl, point_feature], axis=1)
        pcl_p = np.concatenate([pcl, point_probabilities], axis=1)
        pcl_gt = np.concatenate([pcl, gt_map], axis=1)
        pcl_rgb = np.concatenate([pcl, rgb], axis = 1)
        pcl_feature_path = join(pcl_feature_prefix, frame)
        pcl_p_path = join(pcl_p_prefix, frame)
        pcl_gt_path = join(pcl_gt_prefix, frame)
        np.save(pcl_feature_path, pcl_feature)
        np.save(pcl_gt_path, pcl_gt)
        np.save(pcl_p_path, pcl_p)
        # np.savetxt(join(pcl_prefix, frame + '.txt'), pcl_rgb)
        
        
        # points_color = PointCloud(array=pcl, frame_number=0)
        # # point_sem_p = PointCloud(array=pcl, sem_array=semmap, frame_number=0)
        # points_color.save_to_disk(join(sequcence_save_dir, 'Point_Cloud/pc_color', path.split('.')[0]))
        # # point_sem_p.save_to_disk(join(sequcence_save_dir, 'Point_Cloud/pc_sem', path.split('.')[0]))
    f.close()



def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    
    model = tf.estimator.Estimator(
        model_fn=config.model_fn,
        model_dir=FLAGS.model_dir,
        params={
            'output_stride': 16,
            'batch_size': 1,  # Batch size must be 1 because the images' size may differ
            'batch_norm_decay': 0.997,
            'num_classes': _NUM_CLASSES,
            'gpu_id': [0],
            'freeze_batch_norm': False,
            'pretrained_model': None,
            'model': FLAGS.model
        })
    
    DATAPATH = FLAGS.data_dir
    write_S3DIS_lidar_data(DATAPATH, 'train', model)
    write_S3DIS_lidar_data(DATAPATH, 'test', model)
    #divide_multi_sequence()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


