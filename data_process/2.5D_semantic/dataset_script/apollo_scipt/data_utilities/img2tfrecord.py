# -*- coding = utf-8 -*-

from __future__ import absolute_import,division,print_function

import numpy as np
import tensorflow as tf
import time
from PIL import Image
from os import  walk
from os.path import join
import cv2
from labels_apollo import labels
import os
import glob


IMG_HEIGHT = 2710
IMG_WIDTH = 3384
IMG_CHANNELS = 3

NUM_TEST =100
NUM_TOTAL=900
box=(0,0,3328,2688)
Resize_Height=1344
Resize_Width=1664

INSTINCS6 = np.array([[2300.39065314361, 0, 1713.21615190657], [0, 2301.31478860597, 1342.91100799715], [0, 0, 1]])

coormat = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.float32)
for i in range(IMG_HEIGHT):
    for j in range(IMG_WIDTH):
        coormat[i, j] = np.array([j, i, 1], dtype=np.float32)



def disp2pc_mask_map(depthmap,Instrincs,indexmap):
    A = np.linalg.inv(Instrincs)
    A = np.transpose(A)

    height, width = np.shape(depthmap)

    index=np.where(depthmap>300)
    #
    # #temp = eval_time('start')
    #
    depthmap[index]=0

    #temp.minus_last(name='2', processtime_name='depthtime')
    #temp.minus_last(name='3', processtime_name='looptime')

    indexmap=indexmap.reshape([-1,3])
    depth=depthmap.reshape([-1,1])
    pc_map=np.dot(indexmap,A)*depth
    #temp.minus_last(name='3', processtime_name='mattime')
    pc_map = pc_map.reshape([height, width, 3])

    return pc_map



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def walkfilelist(datapath,savepath,num_test,num_total):

    namelist = []
    for i in walk(datapath):
        if i[1]==[]:
            for imgname in i[2]:
                abspath=join(i[0],imgname)
                namelist.append(abspath)
    totalnum=len(namelist)
    print(totalnum)
    idx = np.arange(totalnum)
    np.random.shuffle(idx)
    namelist = np.array(namelist)[idx]
    testid = namelist[0:num_test]
    trainid = namelist[num_test:num_total]
    with open(join(savepath,'trainlist.txt'), 'w') as trainlist:
        with open(join(savepath,'testlist.txt'), 'w') as testlist:
            for trainname in trainid:
                trainlist.write(trainname+'\n')
            for testname in testid:
                testlist.write(testname+'\n')


def Get_applo_list(save_path,phase):
    Exclude_Episode='road01_ins/ColorImage/Record102'
    Camera='Camera 6'
    Name_list_path='/media/luo/Dataset/apollo/Origin_data/name_list/'
    Save_list_path=os.path.join(save_path,phase+'list.txt')
    Trainrecord_path=os.path.join(Savepath,phase+'.tfrecords')
    Abspath='/media/luo/Dataset/apollo/Origin_data/'
    Sample_rate = 3
    def Sample_list(listpath,Sample_rate=Sample_rate):
        sample_list = []
        for l in listpath:
            with open(l) as namelist:
                namelist=namelist.readlines()
                for i in namelist:
                    imgpath = Abspath + i.split('\t')[0]
                    depthpath = imgpath.replace('ColorImage','Depth').replace('.jpg','.png')
                    if os.path.exists(depthpath) and Exclude_Episode not in i and Camera in i:
                        depthmap = np.array(cv2.imread(depthpath, -1), dtype=np.uint16)
                        if np.sum(depthmap==(2**16-1))<(IMG_WIDTH*IMG_HEIGHT*0.8):
                            sample_list.append(i)

                #sample_list=[i for i in sample_list if i.split('/')[2] in episode]
        # idx = np.arange(len(sample_list))
        # np.random.shuffle(idx)
        # sample_list = np.array(sample_list)[idx]

        with open(Save_list_path, 'w') as trainlist:
            for num,name in enumerate(sample_list):
                if num % (Sample_rate)==0:
                    trainlist.write(name)
        with open(Save_list_path, 'r') as trainlist:
            print(len(trainlist.readlines()))


    def write_applo_tfrecord(listpath,savepath):
        with open(listpath) as namelist:
            imglist = [line.rstrip() for line in namelist]
            writer = tf.python_io.TFRecordWriter(savepath)
            num=0
            for name in imglist:
                num+=1
                if num%50==0:
                    print('precess ',num,'/',len(imglist))
                imgpath=Abspath+name.split('\t')[0]
                labelpath=Abspath+name.split('\t')[1]
                # depthpath=imgpath.replace('ColorImage','Depth').replace('jpg','png')
                # depthmap = np.array(cv2.imread(depthpath, -1), dtype=np.uint16)
                # depthmap = np.float32(depthmap) / 200.0
                img = np.array(Image.open(imgpath),dtype=np.uint8)
                labelmap =np.array(Image.open(labelpath),dtype=np.uint8)
                for label in labels:
                    index=np.where(labelmap==label.id)
                    labelmap[index]=label.trainId

                #pc_map=disp2pc_mask_map(depthmap,Instrincs=INSTINCS6,indexmap=coormat)
                img=img[:Resize_Height*2,:Resize_Width*2,:]
                labelmap=labelmap[:Resize_Height*2,:Resize_Width*2]
                #pc_map=pc_map[:Resize_Height*2,:Resize_Width*2,:]

                #pc_map=cv2.resize(pc_map,(Resize_Width,Resize_Height),interpolation=cv2.INTER_LINEAR)
                img=cv2.resize(img,(Resize_Width,Resize_Height),interpolation=cv2.INTER_LINEAR)
                labelmap=cv2.resize(labelmap,(Resize_Width,Resize_Height),interpolation=cv2.INTER_NEAREST)


                # cv2.imwrite(os.path.join(Savepath,'image.png'),img)
                # cv2.imwrite(os.path.join(Savepath,'label.png'),labelmap*10)
                # cv2.imwrite(os.path.join(Savepath,'depth.png'),pc_map[:,:,2])
                #pc_map=np.float16(pc_map)

                img_raw = img.tobytes()
                label_raw = labelmap.tobytes()
                # depth_raw = depth.tobytes()
                #pc_raw = pc_map.tobytes()

                example = tf.train.Example(
                    features=tf.train.Features(feature={'label_raw': _bytes_feature(label_raw),
                                                        'image_raw': _bytes_feature(img_raw),
                                                        }))
                writer.write(example.SerializeToString())
            writer.close()



    Train_list=glob.glob(Name_list_path+'*'+phase+'*')
    print(Train_list)
    # Sample_list(Train_list)
    # Sample_list(Trainlist_path.replace('train','val'),1500,Savepath.replace('train','val'))
    # print('convert to tfrecords begin')
    # start_time = time.time()
    write_applo_tfrecord(Save_list_path,Trainrecord_path)
    # write_applo_tfrecord(Savepath.replace('train','val'),Trainrecord_path.replace('train','val'))
    # duration = time.time() - start_time
    # print('convert to tfrecords end , cost %d sec' % duration)
















if __name__ == '__main__':

    if 1:
        # writefilelist(DATAPATH,SAVEPATH,NUM_TEST,NUM_TOTAL)
        Savepath = '/media/luo/Dataset/apollo/Mydata/Tfrecord/img_label_8000'
        Get_applo_list(Savepath,'train')
        #Get_applo_list(Savepath, 'val')


