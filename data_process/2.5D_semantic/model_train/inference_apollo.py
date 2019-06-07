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
import train,config,dataset_util
from utils import preprocessing
import matplotlib.pyplot as plt
import sys


sys.path.append('../../../')
import common
from data_process.generate_dataset import divide_multi_sequence



#需要修改的变量
_DATA_DIR = common.raw_data_path
_EPISODE_LIST = common.train_sequence_list + common.test_sequence_list
_CAMERA = 'Camera 6'
_SAVE_DIR = common.lidardata_path
_SAMPLE_NUM = common.point_num_per_frame
_START_INDEX = 40
_FRAME_NUMBER = common.frame_num_pre_sequence



#一般不用修改,输入图片的尺寸和相机５的内参
IMG_HEIGHT = 2710
IMG_WIDTH = 3384
INSTRINCS_5 = np.array(
    [[2304.54786556982 , 0, 1686.23787612802], [0, 2305.875668062, 1354.98486439791], [0, 0, 1]])
INSTRINCS_6 = np.array(
    [[2300.39065314361, 0, 1713.21615190657], [0, 2301.31478860597, 1342.91100799715], [0, 0, 1]])
if _CAMERA == 'Camera 6':
    INSTRINCS = INSTRINCS_6
elif _CAMERA == 'Camera 5':
    INSTRINCS = INSTRINCS_5


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default= _DATA_DIR,
                    help='The directory containing the image data.')

parser.add_argument('--episode_list', type=str, default= _EPISODE_LIST,
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default=_SAVE_DIR,
                    help='Path to the directory to generate the inference results')

parser.add_argument('--sample_point', type=str, default= _SAMPLE_NUM,
                    help='Path to the directory to generate the inference results')

parser.add_argument('--start_index', type=str, default=_START_INDEX,
                    help='Path to the directory to generate the inference results')

parser.add_argument('--frame_number', type=str, default=_FRAME_NUMBER,
                    help='Path to the directory to generate the inference results')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

_NUM_CLASSES = 22


def decode_extrincs(path, extrincs_list, savepath):
    extrincs = [i for i in extrincs_list if path in i]
    extrincs = extrincs[0].split(' ')[0:-1]
    extrincs = np.array(extrincs, dtype=np.float32).reshape([4, 4])
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    np.savetxt(join(savepath, path), extrincs)
    return extrincs


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


def Image_map2pcl_global(depthmap, gt_semmap, extrincs, sample_points):
    """
    深度图转换成点云，并且去除距离太远的点和动态的物体的点，最后统一输出采样点数的矩阵
    :param depthmap:
    :param gt_semmap:
    :param extrincs:
    :param sample_points:
    :return:
    """
    depthmap = depthmap.reshape([-1])
    gt_semmap = gt_semmap.reshape([-1])
    for label in dataset_util.labels:
        index = np.where(gt_semmap == label.id)
        gt_semmap[index] = label.trainId
    
    index = np.where(np.logical_and(np.logical_and(depthmap < 10000, gt_semmap >= 9), gt_semmap < 255))
    # 这里是修改
    
    index = np.array(list(index))
    index = np.squeeze(index)
    index, _ = sample_data(index, sample_points)
    
    depthmap = depthmap[index]
    indexmap = Indexmap[index]
    gt_semmap = gt_semmap[index]
    depthmap = np.float32(depthmap) / 200.0
    pc_local = np.dot(np.linalg.inv(INSTRINCS), np.transpose(indexmap)) * np.transpose(depthmap)
    pc_local = np.append(pc_local, np.ones((1, pc_local.shape[1])), axis=0)
    pc_global = np.dot(extrincs, pc_local)[0:3]
    pc_global = np.transpose(pc_global)
    return pc_global, gt_semmap, indexmap[:, :2], index


def bilinear_interp_PointWithIndex(items, xy):
    """

    :param items: 3 dim array
    :param xy:    xy_index
    :return:      points_value after bilinear interp
    """
    xy_floor = np.floor(xy)
    x_floor = xy_floor[:, 0]
    y_floor = xy_floor[:, 1]
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
    return valueArray





def main(unused_argv):
    
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    DATAPATH = FLAGS.data_dir
    Episode_list = FLAGS.episode_list
    
    for Episode in Episode_list:
        Sem_prefix = join(DATAPATH, 'Label', Episode, _CAMERA)
        Extrincs_prefix = join(DATAPATH, 'Pose', Episode, _CAMERA)
        RGB_prefix = join(DATAPATH, 'ColorImage', Episode, _CAMERA)
        Depth_prefix = join(DATAPATH, 'Depth', Episode, _CAMERA)
        Save_path = join(FLAGS.output_dir, Episode)
        pcl_feature_prefix = join(Save_path, 'infer_feature')
        if not os.path.isdir(pcl_feature_prefix):
            os.makedirs(pcl_feature_prefix)
        pcl_p_prefix = join(Save_path, 'infer_label')
        if not os.path.isdir(pcl_p_prefix):
            os.makedirs(pcl_p_prefix)
        pcl_gt_prefix = join(Save_path, 'gt')
        if not os.path.isdir(pcl_gt_prefix):
            os.makedirs(pcl_gt_prefix)
    
        shutil.copy('inference_apollo.py', join(Save_path, 'inference.py'))
        shutil.copy('train.py', join(Save_path, 'train.py'))
        shutil.copy('config.py', join(Save_path, 'config.py'))
        shutil.copy('model.py', join(Save_path, 'model.py'))
    
        extrincs_path = join(Extrincs_prefix, 'pose.txt')
        extrincs_list = open(extrincs_path).readlines()
    
    
        #因为中间有的帧没有GT所以要先判断有没有GT，再加入列表中去
        pathlist = listdir(Sem_prefix)
        pathlist = [i for i in pathlist if 'json' in i][FLAGS.start_index : FLAGS.start_index + FLAGS.frame_number + 1]
        RGB_list=[join(RGB_prefix,i.replace('.json', '.jpg')) for i in pathlist]
        if Episode in common.train_sequence_list:
            List_path = common.lidardata_path + '/train.txt'
        elif Episode in common.test_sequence_list:
            List_path = common.lidardata_path + '/test.txt'
        
        with open(List_path,'a+') as fp:
            for line in RGB_list:
                fp.write(line+'\n')
        #一些支持的函数
    
        model = tf.estimator.Estimator(
            model_fn=config.model_fn,
            model_dir=train.MODEL_DIR,
            params={
                'output_stride': FLAGS.output_stride,
                'batch_size': 1,  # Batch size must be 1 because the images' size may differ
                'batch_norm_decay': 0.997,
                'num_classes': _NUM_CLASSES,
                'gpu_num' : 1,
                'freeze_batch_norm' : False
            })
        image_files =read_examples_list(List_path)
        predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=None)

        f = open(os.path.join(Save_path, 'accuracy_record.txt'), 'w')
        for pred_dict, image_path in zip(predictions, image_files):
            #路径规则
            path=image_path.split('/')[-1]
            sem_path = join(Sem_prefix, path.replace('.jpg', '_bin.png'))
            RGB_path = image_path
            Depth_path = join(Depth_prefix, path.replace('.jpg', '.png'))
            extrincs = decode_extrincs(path.split('.')[0], extrincs_list , join(Save_path, 'pose'))
            pcl_feature_path = join(pcl_feature_prefix, path.split('.')[0])
            pcl_p_path = join(pcl_p_prefix, path.split('.')[0])
            pcl_gt_path = join(pcl_gt_prefix, path.split('.')[0])


            #输出和图片
            RGB = np.array(Image.open(RGB_path))
            pred_P=pred_dict['probabilities_origin']
            pred_ID= pred_dict['classes']
            pred_feature = pred_dict['feature_out']
            feature_shape = pred_feature.shape[0:2]
            p_shape = pred_P.shape[0:2]

            #因为全分辨率的feature实在是太占显存了，所以feature返回不是全分辨率，要使用双线性插值来获得
            origin_shape = RGB.shape[0:2]
            ratio_f = (np.array(feature_shape, dtype= np.float32) - 1) / (np.array(origin_shape, dtype = np.float32) - 1)
            ratio_f = np.array([ratio_f[1], ratio_f[0]])

            ratio_p = (np.array(p_shape, dtype=np.float32) - 1) / (np.array(origin_shape, dtype=np.float32) - 1)
            ratio_p = np.array([ratio_p[1], ratio_p[0]])

            #保存预测的图片
            pred_ID = np.repeat(pred_ID,  3, axis=2)
            print(pred_ID.shape)
            pred_ID_PATH=join(Save_path, 'Pictures/Infer/ID')
            if not os.path.isdir(pred_ID_PATH):
                os.makedirs(pred_ID_PATH)
            misc.imsave(join(pred_ID_PATH, path).replace('.jpg', '.png'), np.uint8(pred_ID))

            #获得采样之后的xy_index, 和铺平之后的index，以及采样完之后的点云
            Sem = np.array(cv2.imread(sem_path, -1),dtype = np.uint8)
            Depth = np.array(cv2.imread(Depth_path, -1),dtype=np.uint16)
            pcl, gt_map, index_xy, index = Image_map2pcl_global(Depth, Sem, extrincs, FLAGS.sample_point)
            index_xy = np.array(index_xy, dtype= np.float32)


            #下面提供了feature和概率的获得方式，第一维都是点的数量
            point_feature = bilinear_interp_PointWithIndex(pred_feature, index_xy * ratio_f)
            point_probabilities = bilinear_interp_PointWithIndex(pred_P, index_xy * ratio_p)
            debug_point_feature = np.argmax(point_probabilities, axis = 1)
            total_num = np.sum(np.where(gt_map != 255))
            correct_num = np.sum(np.where(gt_map == debug_point_feature))
            accuracy = correct_num / total_num
            print(accuracy)
            gt_map = np.expand_dims(gt_map, axis = 1)
            f.write(path + '\t' + str(accuracy) + '\n')
            pcl_feature = np.concatenate([pcl, point_feature], axis = 1)
            pcl_p = np.concatenate([pcl, point_probabilities], axis = 1)
            pcl_gt = np.concatenate([pcl, gt_map], axis = 1)
            np.save(pcl_feature_path, pcl_feature)
            np.save(pcl_gt_path, pcl_gt)
            np.save(pcl_p_path, pcl_p)



            # points_color = PointCloud(array=pcl, frame_number=0)
            # # point_sem_p = PointCloud(array=pcl, sem_array=semmap, frame_number=0)
            # points_color.save_to_disk(join(Save_path, 'Point_Cloud/pc_color', path.split('.')[0]))
            # # point_sem_p.save_to_disk(join(Save_path, 'Point_Cloud/pc_sem', path.split('.')[0]))
        f.close()
        
    divide_multi_sequence()
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
    
