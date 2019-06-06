import numpy as np
from carla.sensor import PointCloud
from carla.transform import Translation,Rotation,Scale,Transform
from PIL import Image
from os.path import join,dirname
from os import listdir
import os
import cv2
from scipy import misc




#Auto get sensorlist

#here to change the abspath
DATADIR='/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/Data/pointnet_test/ver1.0_128line_0.1mplace_fov+-30/start_127'
RGB_Dir=join(DATADIR,'Data','RGB')
Sem_Dir=join(DATADIR,'Data','Semantic')
Lidar_Dir=join(DATADIR,'Data','Lidar')
Sem_P_Dir=join(DATADIR,'Data','Infer_sem')

Camera_list=listdir(RGB_Dir)
Lidarlist=listdir(Lidar_Dir)

#Get each frame pcl path

frame_path=listdir(join(Lidar_Dir,Lidarlist[0]))

#Get insrincs dic

global num
num=0

Instrics={}
for cam_name in Camera_list:
    instrincs=np.loadtxt(join(DATADIR,'pose/instrincs',cam_name,'000000'))
    Instrics[cam_name]=instrincs



def show_frustum_pcl(Datadir, Worldpoints,path):

    Semarray = np.zeros(Worldpoints.shape[0], dtype=np.uint8)
    for i, cam_name in enumerate(Camera_list):
        # Convert pcl to every camera uv

        if 'Infer_sem' in Datadir:
            item_path = join(Datadir, cam_name, path.replace('.ply', '.npy'))
            items = np.load(item_path)


        else:
            item_path = join(Datadir, cam_name, path.replace('.ply', '.png'))
            items = Image.open(item_path)
            items = np.array(items, dtype=np.uint8)
        WIDTH = items.shape[1]
        HEIGHT = items.shape[0]

        depthmap=np.zeros([HEIGHT,WIDTH,3],dtype=np.float32)

        # Get path
        extrincs_path = join(DATADIR, 'pose/extrincs', cam_name, path.replace('.ply', ''))

        # [R|T]
        extrincs = np.loadtxt(extrincs_path)
        extrincs = np.linalg.inv(extrincs)
        points = Worldpoints.transpose()
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)
        points = np.dot(extrincs, points)[0:3]
        local_points=points.transpose()
        # k get uv
        point = np.dot(Instrics[cam_name], points) / points[2, :]
        xy = point.transpose()
        xy_round = np.round(xy)

        indexmap = np.logical_and(np.logical_and(np.logical_and(xy_round[:, 0] > 0, xy_round[:, 0] < WIDTH - 1),
                                                 np.logical_and(xy_round[:, 1] > 0, xy_round[:, 1] < HEIGHT - 1)),
                                  points[2, :].transpose() > 0)
        index = np.where(indexmap)

        RGBindex = (np.int32(xy_round[:, 1][index]), np.int32(xy_round[:, 0][index]))

        #depthmap[RGBindex]=local_points[index]
        depth=local_points[index][:,2:3]
        R=256-50*np.log(depth)
        G=50*np.log(depth)
        B=np.zeros_like(depth)
        depth=np.concatenate([R,G,B],axis=1)


        depthmap[RGBindex]=depth
        # depthmap=cv2.resize(depthmap,(512,512),interpolation=cv2.INTER_NEAREST)
        # items=cv2.resize(items,(512,512),interpolation=cv2.INTER_LINEAR)
        depthmap=0.5*items+0.8*depthmap

        save=os.path.join(Datadir.replace('RGB','Lidar_depth'),cam_name)
        if not os.path.isdir(save):
            os.makedirs(save)
        #np.save(os.path.join(save, path.replace('.ply', '')),depthmap)
        misc.imsave(os.path.join(save, path.replace('ply', 'png')),np.uint8(depthmap))


def pcl2depth(Datadir, Worldpoints,path):

    Semarray = np.zeros(Worldpoints.shape[0], dtype=np.uint8)
    for i, cam_name in enumerate(Camera_list):
        # Convert pcl to every camera uv

        if 'Infer_sem' in Datadir:
            item_path = join(Datadir, cam_name, path.replace('.ply', '.npy'))
            items = np.load(item_path)


        else:
            item_path = join(Datadir, cam_name, path.replace('.ply', '.png'))
            items = Image.open(item_path)
            items = np.array(items, dtype=np.uint8)
        WIDTH = items.shape[1]
        HEIGHT = items.shape[0]

        depthmap=np.zeros([HEIGHT,WIDTH,3],dtype=np.float32)

        # Get path
        extrincs_path = join(DATADIR, 'pose/extrincs', cam_name, path.replace('.ply', ''))

        # [R|T]
        extrincs = np.loadtxt(extrincs_path)
        extrincs = np.linalg.inv(extrincs)
        points = Worldpoints.transpose()
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)
        points = np.dot(extrincs, points)[0:3]
        local_points=points.transpose()
        # k get uv
        point = np.dot(Instrics[cam_name], points) / points[2, :]
        xy = point.transpose()
        xy_round = np.round(xy)

        indexmap = np.logical_and(np.logical_and(np.logical_and(xy_round[:, 0] > 0, xy_round[:, 0] < WIDTH - 1),
                                                 np.logical_and(xy_round[:, 1] > 0, xy_round[:, 1] < HEIGHT - 1)),
                                  points[2, :].transpose() > 0)
        index = np.where(indexmap)

        RGBindex = (np.int32(xy_round[:, 1][index]), np.int32(xy_round[:, 0][index]))

        depthmap[RGBindex]=local_points[index]
        save=os.path.join(Datadir.replace('RGB','Lidar_depth'),cam_name)
        if not os.path.isdir(save):
            os.makedirs(save)
        np.save(os.path.join(save, path.replace('.ply', '')),depthmap)

if __name__ == '__main__':

     for fra,path in enumerate(frame_path):
        #process every frame data
        Worldpoints=[]
        for lidar in Lidarlist:
            lidarpath=join(Lidar_Dir,lidar,path)
            worldpoints=np.loadtxt(lidarpath,skiprows=7)
            Worldpoints.append(worldpoints)
        Worldpoints=np.concatenate(Worldpoints,axis=0)

        savepath = join(RGB_Dir.replace('RGB','lidar_color'), path)
        #show_frustum_pcl(RGB_Dir,Worldpoints,path)
        pcl2depth(RGB_Dir, Worldpoints, path)