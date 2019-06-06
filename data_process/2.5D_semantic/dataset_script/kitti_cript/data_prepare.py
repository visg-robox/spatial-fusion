import numpy as np
import cv2
import tensorflow as tf
from utils.labels import labels
import os
from PIL import Image

#path_processing
def getDirAbsList(Dir):
    """
    :param Dir:
    :return: sorted abs path
    """
    relaPath = os.listdir(Dir)
    relaPath.sort()
    return [os.path.join(Dir, i) for i in relaPath]

def getDirList(Dir):
    """
    :param Dir:
    :return: sorted related path
    """
    relaPath = os.listdir(Dir)
    relaPath.sort()
    return relaPath

def getTextPathList(textfile):
    """
    Get linelist of textfile
    :param textfile:
    :return:
    """
    with open(textfile) as namelist:
        imglist = [line.rstrip() for line in namelist]
    return imglist

def id2trainid(id_img, label_tuple):
    """

    :param id_img: id semanti img uint8
    :param label_tuple: label format
    :return:  trainid semantic img uint8
    """
    trainid_img = np.zeros_like(id_img,dtype= np.uint8)
    for label in label_tuple:
        index = np.where(id_img == label.id)
        trainid_img[index] = label.trainId
    return trainid_img

def writeIterToText(iter, filepath):
    with open(filepath,'w+') as a:
        for i in iter:
            a.write(i + '\n')

def batch2trainid_and_save(input_path_list, output_path_list, label_tuple):
    """
    Have been tested batch convert to trainid and save
    :param input_path_list:
    :param output_path_list:
    :param label_tuple:
    :return:
    """
    for i in range(len(input_path_list)):
        img = cv2.imread(input_path_list[i],flags = -1)
        train_id_img = id2trainid(img, label_tuple)
        cv2.imwrite(output_path_list[i], train_id_img)

def kitti2train_id():
    semantic_dir = '../trainval_data/data_semantics/training/semantic'
    semantic_trainid_Dir = '../trainval_data/data_semantics/training/semantic_trainid'
    path = getDirList(semantic_dir)
    path1 = list(map(lambda i: os.path.join(semantic_dir, i), path))
    path2 = list(map(lambda i : os.path.join(semantic_trainid_Dir, i), path))
    batch2trainid_and_save(path1, path2, labels)



def getTrainValTxtList():
    total_path = '../trainval_data/data_semantics/training/image_2'
    total_list = getDirList(total_path)
    test_txt = '../trainval_data/select_dataset'
    test_list = getTextPathList(test_txt)
    test_list = list(map(lambda i : i.split(' ')[0] + '.png', test_list))
    train_list = set(total_list) - set(test_list)
    imgDir = total_path
    gtDir = total_path.replace('image_2','semantic_trainid')
    train_list = set(map(lambda i : os.path.join(imgDir, i) +'\t' + os.path.join(gtDir, i),train_list))
    test_list = set(map(lambda i : os.path.join(imgDir, i) +'\t' + os.path.join(gtDir, i),test_list))
    traintxt_path ='../trainval_data/train_list.txt'
    valtxt_path ='../trainval_data/val_list.txt'
    writeIterToText(train_list, traintxt_path)
    writeIterToText(test_list, valtxt_path)



def writetfrecord(imglist, gtlist, img_shape, savepath):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    writer = tf.python_io.TFRecordWriter(savepath)
    length = len(imglist)
    h, w = img_shape
    if len(imglist) != len(gtlist):
        raise Exception('imglist len must be same as gtlist')
    total = []
    for i in range(length):
        if(i % 30 == 0):
            print('process ', i , '/30')
        img_r = Image.open(imglist[i])
        img_r = np.array(img_r,dtype = np.uint8)
        r_h, r_w = np.shape(img_r)[0:2]
        label_r = cv2.imread(gtlist[i], -1)
        img = np.zeros(dtype= np.uint8, shape = [h, w, 3])
        label = np.full(dtype = np.uint8, shape = [h, w], fill_value= 255)
        c_h = min(h, r_h)
        c_w = min(w, r_w)
        img[0:c_h, 0:c_w, :] = img_r[0:c_h, 0:c_w, :]
        label[0:c_h, 0:c_w] = label_r[0:c_h, 0:c_w]
        imgmean=np.mean(np.mean(np.array(img),axis=0),axis=0)
        total.append(imgmean)
        img_raw = img.tobytes()

        label_raw = label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={'label_raw': _bytes_feature(label_raw),
                                                                       'image_raw': _bytes_feature(img_raw),
                                                                       }))
        writer.write(example.SerializeToString())
    writer.close()
    return np.mean(np.array(total),axis=0)

def writeKittiTrrecord():
    def list2tfrecord(phase):
        listTxt = '../trainval_data/' + phase + '_list.txt'
        List = getTextPathList(listTxt)
        ImgList = list(map(lambda i: i.split('\t')[0], List))
        gtList = list(map(lambda i: i.split('\t')[1], List))
        Savepath = '../trainval_data/tfrecord'
        shape = [375, 1242]
        Savepath = os.path.join(Savepath, phase +'.tfrecords')
        writetfrecord(ImgList, gtList, shape, Savepath)

    bgr_mean = list2tfrecord('train')
    print('bgr mean :' , bgr_mean)
    list2tfrecord('val')







if __name__ == '__main__':
    writeKittiTrrecord()







