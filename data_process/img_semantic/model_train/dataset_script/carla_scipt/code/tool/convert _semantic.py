# from PIL import Image
import os
import numpy as np
import cv2
from scipy import misc
from PIL import Image
INDIR='/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/out_lidar_camera_semantic/episode_0000/Data/Semantic'


#cam_list=os.listdir(INDIR)

NEW_classes = {
    0: [255, 255, 255],  # None
    1: [220, 20, 60],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [0, 0, 0],  # Other
    4: [70, 70, 70],  # Pedestrians
    5: [0, 255, 255],  # Poles
    6: [255, 255, 0],  # RoadLines
    7: [0, 0, 255],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 35],  # Vegetation
    10: [151, 115, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [255, 124, 0]  # TrafficSigns
}



ZHANG_classes = {
    0: [0, 0, 0],  # None
    1: [220, 20, 60],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [72, 0, 90],  # Other
    4: [70, 70, 70],  # Pedestrians
    5: [153, 153, 153],  # Poles
    6: [157, 234, 50],  # RoadLines
    7: [0, 0, 255],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 30],  # Vegetation
    10: [255, 0, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0]  # TrafficSigns
}




def convert_id(img):
    classes=ZHANG_classes

    result = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(np.sum(img,axis=2) == np.sum(value))] = key
        print(np.sum(value))
    result=np.repeat(result,3,axis=2)
    return result







def convert_color(inputdir,outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    def labels_to_cityscapes_palette(image):
        """
        Convert an image containing CARLA semantic segmentation labels to
        Cityscapes palette.
        """
        classes=ZHANG_classes 
        result =np.zeros((img.shape[0], img.shape[1], 3),dtype=np.uint8)
        for key, value in classes.items():
            result[np.where(img == key)] = value
        return result


    for name in os.listdir(inputdir):
        if 'g' in name and 'color' not in name:
            img=cv2.imread(os.path.join(inputdir,name),-1)
            img=np.array(img,dtype=np.uint8)[:,:,2]
            img=labels_to_cityscapes_palette(img)


            misc.imsave((os.path.join(outdir,name.replace('.','_color.').replace('jpg','png'))),np.uint8(img))

if __name__ == '__main__':



    if 0:
        for cam in cam_list:
            Dir=os.path.join(INDIR,cam)
            convert_color(Dir,Dir)

    if 1:
        Dir='/media/luo/Dataset/CARLA/[divide_train][ICNET_BN] [RNNTEst]/[episode19][feature_map]/Infer_sem/Camera_B0'
        convert_color(Dir,os.path.join(Dir,'RGB_visual'))
        # Dir=Dir.replace('infer','merge')
        # convert_color(Dir,Dir)
        # Dir = Dir.replace('infer', 'label')
        # convert_color(Dir,Dir)


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




