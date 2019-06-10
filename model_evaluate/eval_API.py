import argparse
import math
import numpy as np
#import tensorflow as tf
import socket

import os
import sys
import xlwt
import time
from common import dataset_class_config


APOLLO_CLASSNAME=[' sky ' ,
           ' car ' ,
           ' motorbicycle ' ,
           ' bicycle ' ,
           ' person ' ,
           ' rider ' ,
           ' truck ' ,
           ' bus ' ,
           ' tricycle ' ,
           ' road ' ,
           ' siderwalk ' ,
           ' traffic_cone ' ,
           ' road_pile ' ,
           ' fence ' ,
           ' traffic_light ' ,
           ' pole ' ,
           ' traffic_sign ' ,
           ' wall ' ,
           ' dustbin ' ,
           ' billboard ' ,
           ' building ' ,
           ' vegatation ']

APOLLO_ID_COLOR = [( 70, 130, 180),
                (  0,   0, 142),
                (  0,   0, 230),
                (119,  11,  32),
                (  0, 128, 192),
                (128,  64, 128),
                (128,   0, 192),
                (192,  0,  64),
                (128, 128, 192),
                (192, 128, 192),
                (192, 128,  64),
                (  0,   0,  64),
                (  0,   0, 192),
                ( 64,  64, 128),
                (192,  64, 128),
                (192, 128, 128),
                (  0,  64,  64),
                (192, 192, 128),
                ( 64,   0, 192),
                (192,   0, 192),
                (192,   0, 128),
                (128, 128,  64)]

CARLA_CLASSNAME = [
' road ' ,
' siderwalk ' ,
' traffic_cone ' ,
' road_pile ' ,
' fence ' ,
' traffic_light ' ,
' pole ' ,
' traffic_sign ' ,
' wall ' ,
' dustbin ' ,
' billboard ' ,
' building ' ,
' vegatation ']
CARLA_ID_COLOR =  {
    0: [192 ,128, 192],  # None
    1: [192 ,128,  64],  # Buildings
    2: [  0 ,  0, 64],  # Fences
    3: [  0 ,  0, 192],  # Other
    4: [ 64 ,64 ,128],  # Pedestrians
    5: [192 ,64 ,128],  # Poles
    6: [192 ,128, 128],  # RoadLines
    7: [  0 , 64,  64],  # Roads
    8: [192 ,192, 128],  # Sidewalks
    9: [ 64 ,  0 ,192],  # Vegetation
    10:[192 ,  0, 192],  # Vehicles
    11:[192 ,  0 ,128],  # Walls
    12:[128 ,128, 64],
    255:[0  ,0  , 0] # TrafficSigns
}

S3DIS_CLASSNAME = ['chair', 'ceiling', 'column', 'table', 'window',  'sofa', 'wall', 'floor', 'board', 'door', 'bookcase', 'clutter', 'beam']
S3DIS_ID_COLOR = [
    (255, 255, 255),
    (220, 20, 60),
    (190, 153, 153),
    (0, 0, 0),
    (70, 70, 70),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 255),
    (244, 35, 232),
    (107, 142, 35),
     (151, 115, 255),
     (102, 102, 156),
     (255, 124, 0)]

if dataset_class_config == 'carla':
    CLASSNAME = CARLA_CLASSNAME
    ID_COLOR = CARLA_ID_COLOR
elif dataset_class_config == 'apollo':
    CLASSNAME = APOLLO_CLASSNAME
    ID_COLOR = APOLLO_ID_COLOR
elif dataset_class_config == 'S3DIS':
    CLASSNAME = S3DIS_CLASSNAME
    ID_COLOR = S3DIS_ID_COLOR

VALIDCLASS=len(CLASSNAME)

def log_string(logfile,out_str):
    logfile.write(out_str+'\n')
    logfile.flush()
    print(out_str)


def create_loss_acc(img_pred, output, label, num_classes, ignore_label=-1,valid_classes=-1):
    raw_pred = tf.reshape(output, [-1, num_classes])
    gt = tf.reshape(label, [-1, ])
    raw_output_up = tf.cast(tf.argmax(raw_pred,axis=1),tf.int32)
    img_Pred = tf.argmax(tf.reshape(img_pred, [-1,num_classes]), axis=1)
    img_Pred = tf.cast(img_Pred, dtype=tf.int32)
    #indices = tf.squeeze(tf.where(tf.not_equal(label,ignore_label)),1)
    # valid_num=tf.shape(indices)[0]
    gt = tf.cast(gt, tf.int32)
    #pred = tf.gather(raw_pred, indices)
    # raw_output_up=tf.gather(raw_output_up, indices)
    # correct=tf.equal(raw_output_up,gt)
    # correct_num=tf.shape(tf.where(correct))[0]
    # accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=raw_pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)
    return reduced_loss,img_Pred,raw_output_up,gt


def getaccuracy(pred,gt,validclass=VALIDCLASS,ignoreclass=-1):
    class_num=np.zeros([validclass,3],dtype=np.float32)
    for i in range(validclass):
        class_num[i,0] = np.sum(np.logical_and(np.logical_and(np.equal(pred, i), np.less_equal(gt,validclass)),np.greater_equal(gt, 0)))
        class_num[i,1] = np.sum(np.logical_and(np.equal(gt, i), np.equal(gt, pred)))
        class_num[i,2] = np.sum(np.equal(gt, i))
    return class_num


def eval_print_save(total_accuracy, method_name, logdir, classname = CLASSNAME, record_valid = True):
    per_class_accuracy = total_accuracy[:, 1] / total_accuracy[:, 2]
    mean_accuracy = np.sum(total_accuracy[:, 1]) / np.sum(total_accuracy[:, 2])
    per_class_iou = total_accuracy[:, 1] / (total_accuracy[:, 0] + total_accuracy[:, 2] - total_accuracy[:, 1])
    index = np.where(np.greater(total_accuracy[:, 2], 0))
    new_iou = per_class_iou[index]
    miou = np.mean(new_iou)
    if record_valid:
        classname = np.array(classname)
        classname = classname[index]
        per_class_accuracy = per_class_accuracy[index]
        per_class_iou = new_iou
        data_distribute = total_accuracy[:,2][index]

    #classname=['sky','Building','Road','Sidewalk','Fence','Vegetation','Pole','Car','Traffic Sign','Pedestrian','Bicycle','Lanemarking']
    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    booksheet.write(1,0,'accuracy')
    booksheet.write(1,1, method_name)
    booksheet.write(4, 0, 'iou')
    booksheet.write(4, 1, method_name)
    booksheet.write(7, 0, 'data_distribution')
    for i in range(len(classname)):
        booksheet.write(0, i+3, classname[i])
        booksheet.write(1, i+3, format(per_class_accuracy[i],'.3%'))
        booksheet.write(4, i + 3,format(per_class_iou[i],'.3%'))
        booksheet.write(7, i + 3, format(data_distribute[i]/np.sum(data_distribute),'.3%'))
    booksheet.write(0,2,'mean')
    booksheet.write(1, 2 ,format(mean_accuracy,'.3%'))
    booksheet.write(4, 2  , format(miou,'.3%'))

    workbook.save(logdir+'/'+method_name+'_xlwt.xls')

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logfile=os.path.join(logdir,'log.txt')
    logfile=open(logfile,'a')
    log_string(logfile,'=============%s=============='%method_name)
    log_string(logfile,'perclass accuracy:')
    log_string(logfile,str(per_class_accuracy))
    log_string(logfile,'total_accuracy: %f' % (mean_accuracy))
    log_string(logfile,'perclass iou:')
    log_string(logfile,str(per_class_iou))
    log_string(logfile,'miou: %f' % (miou))
    log_string(logfile,'')
    logfile.close()
    np.savetxt(logdir + '/confusion_matrix.txt', total_accuracy)


class eval_time(object):
    def __init__(self,name):
        self.last_time=0
        self.time_dict={}
        self.process_time_dict={}
        self.add_dict(name)


    def add_dict(self,name):
        cur_time=time.time()
        self.last_time=cur_time
        self.time_dict[name]=cur_time

    def minus_last(self,name,processtime_name):
        cur_time = time.time()
        process_time=cur_time-self.last_time
        self.last_time=cur_time
        self.time_dict[name]=cur_time
        self.process_time_dict[processtime_name]=process_time
        return process_time

    def get_process_time(self,time_new,time_old,processtime_name):
        process_time=self.time_dict[time_new]-self.time_dict[time_old]
        self.process_time_dict[processtime_name]=process_time
        return process_time


    def get_process_time_dif(self,time_new,time_old,processtime_name):
        process_time=self.process_time_dict[time_new]-self.process_time_dict[time_old]
        self.process_time_dict[processtime_name]=process_time
        return process_time

    def printall(self):
        if self.process_time_dict:
            for name in self.process_time_dict:
                print(name,':',self.process_time_dict[name])

        else:
            print('No time recoded')



