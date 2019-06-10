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

IMG_MEAN = np.array((61.90575142,69.36021519,84.3502593 ), dtype=np.float32)
num_classes = 13

model_train30k = './model/icnet_cityscapes_train_30k.npy'
model_trainval90k = './model/icnet_cityscapes_trainval_90k.npy'
model_train30k_bn = './model/icnet_cityscapes_train_30k_bnnomerge.npy'
model_trainval90k_bn = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'

IMG_PATH='/media/luo/Dataset/CARLA/[divide_train][ICNET_BN] [RNNTEst]/[episode19][feature_map]/Data/RGB'
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

def calculate_time(sess, net,feed_dict):
    start = time.time()
    sess.run(net.layers['data'],feed_dict=feed_dict)
    data_time = time.time() - start

    start = time.time()
    sess.run(net.layers['conv6_cls'],feed_dict=feed_dict)
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
        new_h = (int(ori_h/16) + 1) * 16
        new_w = (int(ori_w/16) + 1) * 16
        shape = [new_h, new_w]

        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)
        
        print('Image shape cannot divided by 16, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape

def main():
    args = get_arguments()
    shape=[1024,1024,3]
    x = tf.placeholder(dtype=tf.float32, shape=shape)
    #imgshape=tf.placeholder(dtype=tf.int32,shape=[3])
    #x = tf.placeholder(dtype=tf.float32, shape=imgshape)
    img_tf = preprocess(x)
    img_tf, n_shape = check_input(img_tf)

    # Create network.
    if 1:
        with tf.variable_scope('ICNET'):
            net = ICNet_BN({'data': img_tf}, is_training=False,num_classes=num_classes)
    # elif args.model == 'others':
    #     net = ICNet_BN({'data': img_tf}, num_classes=num_classes)
    # else:
    #     net = ICNet({'data': img_tf}, num_classes=num_classes)
    
    raw_output = tf.nn.softmax(net.layers['conv6_cls'])
    feature_output=net.layers['sub12_sum_interp']
    # Predictions.
    out_feature=tf.image.resize_bilinear(feature_output, size=n_shape, align_corners=True)
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

    camera_dir_list=os.listdir(args.img_path)
    for camera_dir in camera_dir_list:
        img_path=os.path.join(args.img_path,camera_dir)
        if not os.path.exists(img_path.replace('RGB','Infer_sem')):
            os.makedirs(img_path.replace('RGB','Infer_sem'))
        if not os.path.exists(img_path.replace('RGB', 'Feature_map')):
            os.makedirs(img_path.replace('RGB', 'Feature_map'))
        pathlist=image_reader.get_filename(img_path)

        for imgpath in pathlist[:50]+pathlist[-10:]:
            img, filename = load_img(imgpath)
            pred_ID,features = sess.run([raw_output_ID[0],out_feature[0]],feed_dict={x: img})
            pred_ID=np.repeat(np.expand_dims(pred_ID,axis=2),3,axis=2)

            misc.imsave(imgpath.replace('RGB','Infer_sem'),np.uint8(pred_ID))
            np.save(imgpath.replace('RGB','Feature_map').replace('.png','.npy'),features)

if __name__ == '__main__':
    main()
