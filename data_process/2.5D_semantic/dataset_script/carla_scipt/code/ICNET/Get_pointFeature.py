from __future__ import print_function

import argparse
import os
import sys
import time
from PIL import Image
import image_reader
import tensorflow as tf
import numpy as np
from scipy import misc
import shutil

from model import ICNet_BN
from tools import decode_labels

IMG_MEAN = np.array((61.90575142, 69.36021519, 84.3502593), dtype=np.float32)
num_classes = 13

model_train30k = './model/icnet_cityscapes_train_30k.npy'
model_trainval90k = './model/icnet_cityscapes_trainval_90k.npy'
model_train30k_bn = './model/icnet_cityscapes_train_30k_bnnomerge.npy'
model_trainval90k_bn = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'

IMG_PATH = '/media/luo/Dataset/CARLA/[divide_train][ICNET_BN] [RNNTEst]/[episode19][feature_map]/Data/RGB'
snapshot_dir = './model/CRALA_episode19'
SAVE_DIR = './temp/test'


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default=IMG_PATH,
                        help="Path to the RGB image file.",
                        required=False)
    parser.add_argument("--model", type=str, default='others',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        required=False)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")

    return parser.parse_args()


def calculate_time(sess, net, feed_dict):
    start = time.time()
    sess.run(net.layers['data'], feed_dict=feed_dict)
    data_time = time.time() - start

    start = time.time()
    sess.run(net.layers['conv6_cls'], feed_dict=feed_dict)
    total_time = time.time() - start

    inference_time = total_time - data_time

    print('inference time: {}'.format(inference_time))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def load_img(img_path):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    img = misc.imread(img_path, mode='RGB')
    print('input image shape: ', img.shape)

    return img, filename


def preprocess(img):
    # Convert RGB to BGR
    # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    # img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    img = tf.expand_dims(img, dim=0)

    return img


def check_input(img):
    ori_h, ori_w = img.get_shape().as_list()[1:3]

    if ori_h % 16 != 0 or ori_w % 16 != 0:
        new_h = (int(ori_h / 16) + 1) * 16
        new_w = (int(ori_w / 16) + 1) * 16
        shape = [new_h, new_w]

        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)

        print('Image shape cannot divided by 16, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape


def Get_bilinearInterp_value_of_PointWithIndex(items,xy):
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

def Get_around_value_of_PointWithIndex(items,xy):
    Index=(np.int32(xy[:,1]),np.int32(xy[:,0]))
    return items[Index]


def main():
    args = get_arguments()
    shape = [1024, 1024, 3]
    x = tf.placeholder(dtype=tf.float32, shape=shape)
    # imgshape=tf.placeholder(dtype=tf.int32,shape=[3])
    # x = tf.placeholder(dtype=tf.float32, shape=imgshape)
    img_tf = preprocess(x)
    img_tf, n_shape = check_input(img_tf)

    # Create network.
    if 1:
        with tf.variable_scope('ICNET'):
            net = ICNet_BN({'data': img_tf}, is_training=False, num_classes=num_classes)
    # elif args.model == 'others':
    #     net = ICNet_BN({'data': img_tf}, num_classes=num_classes)
    # else:
    #     net = ICNet({'data': img_tf}, num_classes=num_classes)

    raw_output = tf.nn.softmax(net.layers['conv6_cls'])
    feature_output = net.layers['sub12_sum_interp']
    # Predictions.
    out_feature = tf.image.resize_bilinear(feature_output, size=n_shape, align_corners=True)
    raw_output_up = tf.image.resize_bilinear(raw_output, size=n_shape, align_corners=True)
    raw_output_P = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
    print(raw_output_P)
    raw_output_ID = tf.argmax(raw_output_P, dimension=3)
    pred = decode_labels(raw_output_ID, shape, num_classes)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    if args.model == 'train':
        print('Restore from train30k model...')
        net.load(model_train30k, sess)
    elif args.model == 'trainval':
        print('Restore from trainval90k model...')
        net.load(model_trainval90k, sess)
    elif args.model == 'train_bn':
        print('Restore from train30k bnnomerge model...')
        net.load(model_train30k_bn, sess)
    elif args.model == 'trainval_bn':
        print('Restore from trainval90k bnnomerge model...')
        net.load(model_trainval90k_bn, sess)
    else:
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=var_list)
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            load(loader, sess, ckpt.model_checkpoint_path)

    camera_dir_list = os.listdir(args.img_path)
    frame_dir = os.path.join(args.img_path, camera_dir_list[0])
    frame_list=os.listdir(frame_dir)

    for frame in frame_list:
        Points_feature=[]

        for camera_dir in camera_dir_list:
            img_dir=os.path.join(args.img_path,camera_dir)
            img_path=os.path.join(img_dir,frame)
            img, filename = load_img(img_path)


            pointsIndex_path=img_path.replace('RGB','img_index').replace('png','npy')
            pointsIndex=np.load(pointsIndex_path)
            xyz=pointsIndex[:,:3]
            xy_index=pointsIndex[:,3:]

            pred_ID, features = sess.run([raw_output_ID[0], out_feature[0]], feed_dict={x: img})
            pred_ID = np.repeat(np.expand_dims(pred_ID, axis=2), 3, axis=2)

            points_feature=Get_bilinearInterp_value_of_PointWithIndex(features,xy_index)
            points_feature=np.concatenate([xyz,points_feature],axis=1)
            Points_feature.append(points_feature)


            Infer_dir = img_dir.replace('RGB', 'Infer_sem')
            if not os.path.exists(Infer_dir):
                os.makedirs(Infer_dir)
            feature_dir = args.img_path.replace('RGB', 'points_feature')
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)

            misc.imsave(img_path.replace('RGB', 'Infer_sem'), np.uint8(pred_ID))

        Points_feature=np.concatenate(Points_feature,axis=0)
        np.save(os.path.join(feature_dir,frame.replace('.png','')), np.float32(Points_feature))
        pass

if __name__ == '__main__':
    main()
