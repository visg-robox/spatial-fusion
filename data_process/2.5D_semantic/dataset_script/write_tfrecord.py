# -*- coding = utf-8 -*-

from __future__ import absolute_import,division,print_function

import numpy as np
import tensorflow as tf
import time
from PIL import Image
from os import  walk
from os.path import join

import os
import glob






def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



#here to change
def decode_img_S3DIS(img_path):
    """
    :param img_path:
    :return: decode img map RGB uint8
    """
    img = np.array(Image.open(img_path), dtype=np.uint8)
    return  img

def decode_gt_S3DIS(label_path):
    """
    :param label_path: path of a single label file
    :return: a decoded label map, should be ID matrix Uint8
    """
    label_map =  np.array(Image.open(label_path), dtype=np.uint8)

    def get_index(color):
        ''' Parse a color as a base-256 number and returns the index
        Args:
            color: A 3-tuple in RGB-order where each element \in [0, 255]
        Returns:
            index: an int containing the indec specified in 'color'
        '''
        return color[0] * 256 * 256 + color[1] * 256 + color[2]
    
    label_ID_map = label_map[:,:,0] * 256 * 256 + label_map[:,:,0] * 256 + label_map[:,:,0]
    print(label_ID_map)
    
    return label_map
#here to change




def write_tfrecord(img_path_list, label_path_list, savepath):
    writer = tf.python_io.TFRecordWriter(savepath)
    num=0
    for i in range(len(img_path_list)):
        num+=1
        if num%50==0:
            print('precess ',num,'/',len(img_path_list))
        img_path=img_path_list[i]
        label_path=label_path_list[i]

        assert img_path.split('/')[-1].split('_')[:-1] == label_path.split('/')[-1].split('_')[:-2]

        img = decode_img_S3DIS(img_path)
        img = np.array(img,dtype=np.uint8)
        
        label_map = decode_gt_S3DIS(label_path)
        label_map =np.array(label_map,dtype=np.uint8)
        
        img_raw = img.tobytes()
        label_raw = label_map.tobytes()

        example = tf.train.Example(
            features=tf.train.Features(feature={'label_raw': _bytes_feature(label_raw),
                                                'image_raw': _bytes_feature(img_raw),
                                                }))
        writer.write(example.SerializeToString())
    writer.close()




if __name__ == '__main__':
    #here to change
    train_data_path = '/data1/3d_map/data/S3DIS/train/'
    valid_data_path = '/data1/3d_map/data/S3DIS/test/'

    train_img_list = sorted(glob.glob(train_data_path + 'area*/data/rgb/*.png'))
    train_label_list = sorted(glob.glob(train_data_path + 'area*/data/semantic_pretty/*.png'))

    val_img_list = sorted(glob.glob(valid_data_path + 'area*/data/rgb/*.png'))
    val_label_list = sorted(glob.glob(valid_data_path + 'area*/data/semantic_pretty/*.png'))
    save_path = '/data1/3d_map/data/S3DIS/'

    #here to change
    
    
    write_tfrecord(train_img_list, train_label_list, os.path.join(save_path, 'train.tfrecords'))
    write_tfrecord(val_img_list, val_label_list, os.path.join(save_path, 'val.tfrecords'))
    



