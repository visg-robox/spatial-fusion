import argparse
import math
import numpy as np
import tensorflow as tf
import socket

import os
import sys
import xlwt
import time


CLASSNAME=['void','Buildings','Fences','Other','Pedestrians','Poles','RoadLines','Roads','Sidewalks','Vegetation','Vehicles','Walls','TrafficSigns']
VALIDCLASS=len(CLASSNAME)
global MIOU
MIOU=0



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
        class_num[i,0] = np.sum(np.logical_and(np.equal(pred, i),np.not_equal(gt,ignoreclass)))
        class_num[i,1] = np.sum(np.logical_and(np.equal(gt, i), np.equal(gt, pred)))
        class_num[i,2] = np.sum(np.equal(gt, i))
    return class_num


def eval_print_save(total_accuracy,method_name,logdir,classname,Forcerecord=True):
    per_class_accuracy = total_accuracy[:, 1] / total_accuracy[:, 2]
    mean_accuracy = np.sum(total_accuracy[:, 1]) / np.sum(total_accuracy[:, 2])
    per_class_iou = total_accuracy[:, 1] / (total_accuracy[:, 0] + total_accuracy[:, 2] - total_accuracy[:, 1])
    miou = np.mean(per_class_iou)
    global MIOU
    if Forcerecord:
        MIOU=0
    if miou>MIOU:
        MIOU=miou
    #classname=['sky','Building','Road','Sidewalk','Fence','Vegetation','Pole','Car','Traffic Sign','Pedestrian','Bicycle','Lanemarking']
        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
        booksheet.write(1,0,'accuracy')
        booksheet.write(1,1, method_name)

        booksheet.write(4, 0, 'iou')
        booksheet.write(4, 1, method_name)

        booksheet.write(7, 0, 'data_distribution')
        for i in range(classname.shape[0]):
            booksheet.write(0, i+3, classname[i])
            booksheet.write(1, i+3, format(per_class_accuracy[i],'.3%'))
            booksheet.write(4, i + 3,format(per_class_iou[i],'.3%'))
            booksheet.write(7, i + 3, format(total_accuracy[i,2]/np.sum(total_accuracy[:, 2]),'.3%'))
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



