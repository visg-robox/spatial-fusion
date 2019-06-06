import numpy as np
from carla.sensor import PointCloud
from carla.transform import Translation,Rotation,Scale,Transform
from PIL import Image
from os.path import join,dirname
from os import listdir
import os




#Auto get sensorlist

#here to change the abspath
DATADIR='/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/Data/out_lidar_vedio_test/episode_19'
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

def makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def merge_pcl(Datadir,Worldpoints):
    Pointarray = []
    Indexlog = []
    Semarray = np.zeros(Worldpoints.shape[0], dtype=np.uint8)

    for i, cam_name in enumerate(Camera_list):
        # Convert pcl to every camera uv

        if  'Infer_sem' in Datadir:
            if fra%3>0:
                items=np.zeros([1024,1024,13],dtype=np.float32)
                GTpath=join(Datadir, cam_name, path.replace('.ply', '.png').replace('Infer_sem','Semantic'))
                GT=Image.open(GTpath)
                GT=np.array(GT)
                h,w=GT.shape[0:2]
                for i in range(h):
                    for j in range(w):
                        Class=GT[i,j,0]
                        items[i,j,Class]=1
                global num
                num+=1
                print('finish',num)

            else:
                item_path = join(Datadir, cam_name, path.replace('.ply', '.npy'))
                items = np.load(item_path)


        else:
            item_path = join(Datadir, cam_name, path.replace('.ply', '.png'))
            items = Image.open(item_path)
            items = np.array(items, dtype=np.uint8)
        WIDTH = items.shape[1]
        HEIGHT = items.shape[0]

        pointarray = np.zeros([Worldpoints.shape[0], items.shape[2]], dtype=np.float32)
        # Get path
        extrincs_path = join(DATADIR, 'pose/extrincs', cam_name, path.replace('.ply', ''))

        # [R|T]
        extrincs = np.loadtxt(extrincs_path)
        extrincs = np.linalg.inv(extrincs)
        points = Worldpoints.transpose()
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)
        points = np.dot(extrincs, points)[0:3]
        # k get uv
        point = np.dot(Instrics[cam_name], points) / points[2, :]
        xy = point.transpose()
        xy_round = np.round(xy)
        xy_floor = np.floor(xy)
        x_floor = xy_floor[:, 0]
        y_floor = xy_floor[:, 1]
        xy_ceil = np.ceil(xy)
        x_ceil = xy_ceil[:, 0]
        y_ceil = xy_ceil[:, 1]

        # Get valid RGB index
        indexmap = np.logical_and(np.logical_and(np.logical_and(xy_round[:, 0] > 0, xy_round[:, 0] < WIDTH - 1),
                                                 np.logical_and(xy_round[:, 1] > 0, xy_round[:, 1] < HEIGHT - 1)),
                                  points[2, :].transpose() > 0)
        index = np.where(indexmap)

        if  'Sem_color' in Datadir:
            RGBindex = (np.int32(xy_round[:, 1][index]), np.int32(xy_round[:, 0][index]))
            Semarray[index] = items[RGBindex]

        else:
            RGBindex_A = (np.int32(y_ceil[index]), np.int32(x_floor[index]))
            RGBindex_B = (np.int32(y_ceil[index]), np.int32(x_ceil[index]))
            RGBindex_C = (np.int32(y_floor[index]), np.int32(x_ceil[index]))
            RGBindex_D = (np.int32(y_floor[index]), np.int32(x_floor[index]))
            x0 = (xy[:, 0] - x_floor)[index]
            x1 = (x_ceil - xy[:, 0])[index]
            y0 = (xy[:, 1] - y_floor)[index]
            y1 = (y_ceil - xy[:, 1])[index]
            x0 = np.expand_dims(x0, axis=1)
            x1 = np.expand_dims(x1, axis=1)
            y0 = np.expand_dims(y0, axis=1)
            y1 = np.expand_dims(y1, axis=1)
            pointarray[index] = items[RGBindex_A] * x1 * y0 + items[RGBindex_B] * x0 * y0 + items[
                RGBindex_C] * x0 * y1 + items[RGBindex_D] * x1 * y1
            Pointarray.append(np.expand_dims(pointarray, axis=2))
            Indexlog.append(np.expand_dims(indexmap, axis=1))

    if 'Sem_color' in Datadir :
        return Semarray

    Pointarray = np.concatenate(Pointarray, axis=2)
    Indexlog = np.concatenate(Indexlog, axis=1)
    Pointarray = np.sum(Pointarray, axis=2) / (np.sum(Indexlog, axis=1, keepdims=True) + 1e-9)
    return Pointarray

def get_Zhangjian_RNN_LidarGT(infer_lidar_dir,save_gt_dir):
    temp_dir = os.path.join(save_gt_dir, 'infer_npy')
    gt_dir = os.path.join(save_gt_dir, 'gt_npy')
    makedir(temp_dir)
    makedir(gt_dir)

    for i,frame in enumerate(os.listdir(infer_lidar_dir)):
        if i%10==0:
            print('process',i)
        worldpoints_infer = np.loadtxt(os.path.join(infer_lidar_dir,frame), skiprows=7)
        temp_path=os.path.join(temp_dir,frame.replace('.ply',''))
        save_gt_path=os.path.join(gt_dir,frame.replace('.ply',''))

        Worldpoints = worldpoints_infer[:,:3]
        GT_array = np.zeros([Worldpoints.shape[0],3], dtype=np.float32)

        for i, cam_name in enumerate(Camera_list):
            # Convert pcl to every camera uv

            item_path = join(Sem_Dir, cam_name, frame.replace('.ply', '.png'))
            items = Image.open(item_path)
            items = np.array(items, dtype=np.uint8)

            WIDTH = items.shape[1]
            HEIGHT = items.shape[0]

            # Get path
            extrincs_path = join(DATADIR, 'pose/extrincs', cam_name, frame.replace('.ply', ''))

            # [R|T]
            extrincs = np.loadtxt(extrincs_path)
            extrincs = np.linalg.inv(extrincs)
            points = Worldpoints.transpose()
            points = np.append(points, np.ones((1, points.shape[1])), axis=0)
            points = np.dot(extrincs, points)[0:3]
            # k get uv
            point = np.dot(Instrics[cam_name], points) / points[2, :]
            xy = point.transpose()
            xy_round = np.round(xy)


            # Get valid RGB index
            indexmap = np.logical_and(np.logical_and(np.logical_and(xy_round[:, 0] > 0, xy_round[:, 0] < WIDTH - 1),
                                                     np.logical_and(xy_round[:, 1] > 0, xy_round[:, 1] < HEIGHT - 1)),
                                      points[2, :].transpose() > 0)
            index = np.where(indexmap)

            RGBindex = (np.int32(xy_round[:, 1][index]), np.int32(xy_round[:, 0][index]))
            GT_array[index] = items[RGBindex]
        GT_array=GT_array[:,0:1]
        GT_array=np.concatenate([Worldpoints,GT_array],axis=1)
        np.save(temp_path, worldpoints_infer)
        np.save(save_gt_path, GT_array)


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

    if 0:
        for fra,path in enumerate(frame_path):
            #process every frame data
            Worldpoints=[]

            for lidar in Lidarlist:
                lidarpath=join(Lidar_Dir,lidar,path)
                worldpoints=np.loadtxt(lidarpath,skiprows=7)
                Worldpoints.append(worldpoints)
            Worldpoints=np.concatenate(Worldpoints,axis=0)

            savepath = join(RGB_Dir.replace('RGB','lidar_color'), path)
            #pcl2depth(RGB_Dir,Worldpoints,path)

            #save RGB pcl
            if 0:
                pointRGB = merge_pcl(RGB_Dir, Worldpoints)
                points_color=PointCloud(array=Worldpoints, color_array=pointRGB,frame_number=0)
                points_color.save_to_disk(savepath)


            #save semantic color pcl
            if 1:
                pointSem_P = merge_pcl(Sem_P_Dir, Worldpoints)
                points_sem_p = PointCloud(array=Worldpoints, sem_array=pointSem_P, frame_number=0)
                points_sem_p.save_to_disk(savepath.replace('color','sem_p'))

    if 1:
        Infer_dir=os.path.join(DATADIR,'Data','lidar_sem_p')
        Save_dir=os.path.join(DATADIR,'Data','lidar_gt')
        get_Zhangjian_RNN_LidarGT(Infer_dir,Save_dir)













