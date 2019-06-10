
import numpy as np
from carla.sensor import PointCloud
from PIL import Image
import cv2
from os.path import join,dirname
from os import listdir


DATAPATH='/media/luo/temp/apollo'
Episode='Record001/Camera 5'
Sem_prefix=join(DATAPATH,'road02_ins/Label',Episode)
Extrincs_prefix=join(DATAPATH,'road02_ins/Pose',Episode)
RGB_prefix=join(DATAPATH,'road02_ins/ColorImage',Episode)
Depth_prefix=join(DATAPATH,'Depth',Episode)

IMG_HEIGHT=2710
IMG_WIDTH=3384
INSTRINCS=np.array([[2300.39065314361,0,1686.23787612802],[0,2305.875668062,1354.98486439791],[0,0,1]])

extrincs_path=join(Extrincs_prefix,'pose.txt')
extrincs_list=open(extrincs_path).readlines()

Coormat = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.float32)
for i in range(IMG_HEIGHT):
    for j in range(IMG_WIDTH):
        Coormat[i, j] = np.array([j, i, 1], dtype=np.float32)
Indexmap=Coormat.reshape([-1,3])

def decode_extrincs(path):
    extrincs=[i for i in extrincs_list if path in i]
    extrincs=extrincs[0].split(' ')[0:-1]
    extrincs=np.array(extrincs,dtype=np.float32).reshape([4,4])
    np.savetxt(join(DATAPATH,'my_pcl','extrincs',path),extrincs)
    return extrincs


def Image_map2pcl_global(colormap,depthmap,semmap,extrincs):

    colormap=colormap.reshape([-1,3])
    depthmap=depthmap.reshape([-1])
    semmap=semmap.reshape([-1])
    index=np.where(np.logical_and(np.logical_and(depthmap<60000,semmap>=49),semmap<=113))

    colormap=colormap[index]
    depthmap=depthmap[index]
    semmap=semmap[index]
    indexmap=Indexmap[index]
    depthmap=np.float32(depthmap)/200.0

    pc_local=np.dot(np.linalg.inv(INSTRINCS),np.transpose(indexmap))*np.transpose(depthmap)
    pc_local = np.append(pc_local, np.ones((1,pc_local.shape[1])), axis=0)
    pc_global=np.dot(extrincs,pc_local)[0:3]
    pc_global=np.transpose(pc_global)

    return pc_global,colormap,semmap


if __name__ == '__main__':
    pathlist=listdir(Sem_prefix)
    pathlist=[i for i in pathlist if 'json' in i]
    for path in pathlist:
        sem_path=join(Sem_prefix,path.replace('.json','_bin.png'))
        RGB_path=join(RGB_prefix,path.replace('.json','.jpg'))
        Depth_path=join(Depth_prefix,path.replace('.json','.png'))

        extrincs=decode_extrincs(path.split('.')[0])
        RGB=np.array(Image.open(RGB_path))
        Sem=np.array(cv2.imread(sem_path,-1),dtype=np.uint8)
        Depth=np.array(cv2.imread(Depth_path,-1),dtype=np.uint16)

        pcl,colormap,semmap=Image_map2pcl_global(RGB,Depth,Sem,extrincs)

        points_color = PointCloud(array=pcl, color_array=colormap, frame_number=0)
        points_color.save_to_disk(join(DATAPATH,'my_pcl',Episode,path.split('.')[0]))











