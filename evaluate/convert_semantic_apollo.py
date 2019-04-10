# from PIL import Image
import os
import numpy as np
import cv2
from scipy import misc
from PIL import Image
from labels_apollo import labels
import shutil

Num_Class=22
Apollo_label_colours=np.zeros([Num_Class,3],dtype=np.uint8)
Apollo_class={}

for label in labels:
    if label.trainId<=Num_Class:
    # color = (int(label.color[2:4],16),int(label.color[4:6],16),int(label.color[6:8],16))
        color = label.color
        r = color // (256 * 256)
        g = (color - 256 * 256 * r) // 256
        b = (color - 256 * 256 * r - 256 * g)
        Apollo_label_colours[label.trainId]=np.array([r,g,b],dtype=np.uint8)
        print('\'',label.name,'\'',',')
print(Apollo_label_colours)
for i in range(Num_Class):
    Apollo_class[i]=Apollo_label_colours[i]

Classname_apollo=[' sky ' ,
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


def makerdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def cp_RGB(infer_dir,GT_dir,GT_out_dir):
    makerdir(GT_out_dir)
    for frame in os.listdir(infer_dir):
        GT_path=os.path.join(GT_dir,frame.replace('png','jpg'))
        GT_out_path=os.path.join(GT_out_dir,frame.replace('png','jpg'))
        shutil.copy(GT_path,GT_out_path)


def cp_GT(infer_dir,GT_dir,GT_out_dir):
    makerdir(GT_out_dir)
    for frame in os.listdir(infer_dir):
        GT_path=os.path.join(GT_dir,frame.replace('.','_bin.'))
        GT_out_path=os.path.join(GT_out_dir,frame)
        labelmap = np.array(Image.open(GT_path), dtype=np.uint8)
        for label in labels:
            index = np.where(labelmap == label.id)
            labelmap[index] = label.trainId
        labelmap=np.repeat(np.expand_dims(labelmap,axis=2),3,axis=2)
        misc.imsave(GT_out_path,labelmap)

def convert_id(img):
    classes=Apollo_class

    result = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(np.sum(img,axis=2) == np.sum(value))] = key
        print(np.sum(value))
    result=np.repeat(result,3,axis=2)
    return result

def convert_color(inputdir,outdir):
    makerdir(outdir)
    def labels_to_cityscapes_palette(image):
        """
        Convert an image containing CARLA semantic segmentation labels to
        Cityscapes palette.
        """
        classes=Apollo_class
        result =np.zeros((img.shape[0], img.shape[1], 3),dtype=np.uint8)
        for key, value in classes.items():
            result[np.where(img == key)] = value
        return result

    for name in os.listdir(inputdir):
        if 'g' in name and 'color' not in name:
            img=cv2.imread(os.path.join(inputdir,name),-1)
            if len(img.shape)==2:
                img=np.expand_dims(img,axis=2)
            img=np.array(img,dtype=np.uint8)[:,:,0]
            img=labels_to_cityscapes_palette(img)
            misc.imsave((os.path.join(outdir,name.replace('.','_color.').replace('jpg','png'))),np.uint8(img))


def convert_color_ignore(inputdir,ignoredir,savedir):

    makerdir(savedir)
    for name in os.listdir(inputdir):
        if 'g' in name and 'color' not in name:
            inputimg=os.path.join(inputdir,name)
            ignoreimg=os.path.join(ignoredir,name)
            ignoreimg=np.array(Image.open(ignoreimg))
            inputimg=np.array(Image.open(inputimg))
            if len(ignoreimg.shape)==2:
                ignoreimg=np.expand_dims(ignoreimg,axis=2)
            if len(inputimg.shape) == 2:
                inputimg = np.expand_dims(inputimg, axis=2)
                inputimg= np.repeat(inputimg,3,axis=2)
            index=np.where(ignoreimg[:,:,0]==255)
            inputimg[index]=[255,255,255]
            misc.imsave(os.path.join(savedir,name),inputimg)


if __name__ == '__main__':

    INDIR = '/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/out_lidar_camera_semantic/episode_0000/Data/Semantic'

    # #CARLA 批量所以相机转换
    # if 0:
    #     for cam in cam_list:
    #         Dir=os.path.join(INDIR,cam)
    #         convert_color(Dir,Dir)


    frame='Record005/Camera 5' # chage here
    infer_dir=os.path.join('/media/luo/temp/apollo/My_pcl/ver_deeplab',frame,'Pictures/Infer/ID')
    GT_dir=os.path.join('/media/luo/temp/apollo/Origin_data/road02_ins/Label',frame)
    GT_out_dir=os.path.join(infer_dir,'../../GT/ID')
    RGB_dir=GT_dir.replace('Label','ColorImage')
    RGB_out_dir=os.path.join(infer_dir,'../../RGB')
    ignore_skydir = os.path.join(infer_dir, '../nosky_ID')

    if 0:
        cp_RGB(infer_dir,RGB_dir,RGB_out_dir)
        #cp_GT(infer_dir,GT_dir,GT_out_dir)


    if 0:

        convert_color_ignore(infer_dir,GT_out_dir,ignore_skydir)


    #转换目标文件夹下的图片
    if 1:
        Dir='/media/luo/temp/apollo/My_pcl/temp/Record031/Camera 5/Pictures/Infer/ID'
        convert_color(Dir,os.path.join(Dir,'..','RGB_visual'))
        # Dir=Dir.replace('infer','merge')
        # convert_color(Dir,Dir)
        # Dir = Dir.replace('infer', 'label')
        # convert_color(Dir,Dir)

    #转换融合后的并抠图
    if 0:
        imgdir='/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/out_lidar_camera_semantic/episode_17/Data/temp'
        savedir='/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/out_lidar_camera_semantic/episode_17/Eval/v2_F0/merge'

        for i in range(len(imgdir)):
            imgpath=os.path.join(imgdir,'Semantic_Finish_Eva_'+str(i+1)+'.png')
            savepath=os.path.join(savedir,'{:0>6d}.png'.format(i+1))
            infer_path=os.path.join(imgdir.replace('temp','Infer_sem2/Camera_F0'),'{:0>6d}.png'.format(i+1))
            label_path=infer_path.replace('Infer_sem2','Semantic')
            img=Image.open(imgpath)
            img=np.array(img,dtype=np.uint16)
            img=convert_id(img)
            # img=convert_id(img)
            infer_img=np.array(Image.open(infer_path))
            label_img=np.array(Image.open(label_path))
            index=np.where(img[:,:,0]==0)
            infer_img[index]=[0,0,0]
            label_img[index]=[0,0,0]

            misc.imsave(savepath,np.uint8(img))
            misc.imsave(savepath.replace('merge','infer'),np.uint8(infer_img))
            misc.imsave(savepath.replace('merge','label'),np.uint8(label_img))
            # misc.imsave(savepath.replace('merge', 'residual/merge'), np.float32(img[:,:,0]==label_img[:,:,0]))
            # misc.imsave(savepath.replace('merge', 'residual/infer'), np.float32(infer_img[:,:,0]==label_img[:,:,0]))




