
import numpy as np
from PIL import Image
#from carla.sensor import PointCloud
from os.path import join,dirname
from os import listdir
import os
import glob



def Fetch_PointWithImgIndex_of_ProjectedPointcloud(pointcloud,instrincs,extrincs,img_resolution):

    """Project PointCloud to a img plane to get the inlier and their img xy

    :param pointcloud: Nx3 Array
    :param instrincs:  3x3 Instincs matrix
    :param extrincs:   4x4 Extrincs matrix
    :param img_resolution:  [h,w]
    :return: PointWithImgIndex
    """

    height,width=img_resolution
    pointcloud=np.float32(pointcloud)
    xyz = pointcloud.transpose()
    homo_world_points = np.append(xyz, np.ones((1, xyz.shape[1])), axis=0)
    local_points = np.dot(extrincs, homo_world_points)[0:3]
    # k get uv
    camera_xy = np.dot(instrincs,local_points) / local_points[2, :]
    camera_xy = camera_xy.transpose()[:,:2]

    indexmap = np.logical_and(np.logical_and(np.logical_and(camera_xy[:, 0] > 0, camera_xy[:, 0] < width - 1),
                                             np.logical_and(camera_xy[:, 1] > 0, camera_xy[:, 1] < height - 1)),
                              local_points[2, :].transpose() > 0)

    index = np.where(indexmap)

    PointWithImgIndex=np.concatenate([pointcloud[index],camera_xy[index]],axis=1)
    PointWithImgIndex=np.array(PointWithImgIndex,dtype=np.float32)
    return PointWithImgIndex

def bilinear_interp_PointWithIndex(items,xy):
    """

    :param items: 3 dim array
    :param xy:    xy_index
    :return:      points_value after bilinear interp
    """
    xy_floor = np.floor(xy)
    x_floor = xy_floor[:, 0]
    y_floor = xy_floor[:, 1]
    xy_ceil = np.ceil(xy)
    x_ceil = xy_ceil[:, 0]
    y_ceil = xy_ceil[:, 1]
    # Get valid RGB index

    RGBindex_A = (np.int32(y_ceil), np.int32(x_floor))
    RGBindex_B = (np.int32(y_ceil), np.int32(x_ceil))
    RGBindex_C = (np.int32(y_floor), np.int32(x_ceil))
    RGBindex_D = (np.int32(y_floor), np.int32(x_floor))
    x0 = (xy[:, 0] - x_floor)
    x1 = (x_ceil - xy[:, 0])
    y0 = (xy[:, 1] - y_floor)
    y1 = (y_ceil - xy[:, 1])
    x0 = np.expand_dims(x0, axis=1)
    x1 = np.expand_dims(x1, axis=1)
    y0 = np.expand_dims(y0, axis=1)
    y1 = np.expand_dims(y1, axis=1)
    valueArray = items[RGBindex_A] * x1 * y0 + items[RGBindex_B] * x0 * y0 + items[
        RGBindex_C] * x0 * y1 + items[RGBindex_D] * x1 * y1
    return valueArray

def ply_to_PointArray(path,skiprows=7):
    return np.loadtxt(path, skiprows=skiprows)

def Get_around_value_of_PointWithIndex(items,xy):
    Index=(np.int32(xy[:,1]),np.int32(xy[:,0]))
    return items[Index]

def makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def CARLA_projection():
    DATADIR = '/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/Data/out_lidar_vedio_test/episode_19'
    RGB_Dir = join(DATADIR, 'Data', 'RGB')
    Sem_Dir = join(DATADIR, 'Data', 'Semantic')
    Lidar_Dir = join(DATADIR, 'Data', 'Lidar')
    Sem_P_Dir = join(DATADIR, 'Data', 'Infer_sem')
    Camera_list = listdir(RGB_Dir)
    Lidarlist = listdir(Lidar_Dir)
    # Get each frame pcl path
    RESOLUTION=[1024,1024]
    frame_path = listdir(join(Lidar_Dir, Lidarlist[0]))

    Instrics = {}
    for cam_name in Camera_list:
        instrincs = np.loadtxt(join(DATADIR, 'pose/instrincs', cam_name, '000000'))
        Instrics[cam_name] = instrincs

    # Get insrincs dic

    for fra, path in enumerate(frame_path):
        # process every frame data
        Worldpoints = []
        for lidar in Lidarlist:
            lidarpath = join(Lidar_Dir, lidar, path)
            Worldpoints.append(ply_to_PointArray(lidarpath))
        Worldpoints = np.concatenate(Worldpoints, axis=0)

        for camera in Camera_list:
            extrincs_path = join(DATADIR, 'pose/extrincs', camera, path.replace('.ply', ''))
            extrincs = np.loadtxt(extrincs_path)
            extrincs = np.linalg.inv(extrincs)
            instrincs=Instrics[camera]
            PointWithImgIndex=Fetch_PointWithImgIndex_of_ProjectedPointcloud(Worldpoints,instrincs,extrincs,RESOLUTION)
            save_dir=join(RGB_Dir.replace('RGB','img_index'),camera)
            makedir(save_dir)
            savepath = join(save_dir, path.split('.')[0])
            np.save(savepath,PointWithImgIndex)
            pass

def Test_pointWithIndex():
    DATADIR = '/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/Data/out_lidar_vedio_test/episode_19'
    RGB_Dir = join(DATADIR, 'Data', 'RGB')
    Sem_Dir = join(DATADIR, 'Data', 'Semantic')
    Lidar_Dir = join(DATADIR, 'Data', 'Lidar')
    Sem_P_Dir = join(DATADIR, 'Data', 'Infer_sem')
    Camera_list = listdir(RGB_Dir)
    Lidarlist = listdir(Lidar_Dir)
    # Get each frame pcl path
    RESOLUTION = [1024, 1024]
    frame_path = listdir(join(Lidar_Dir, Lidarlist[0]))

    Instrics = {}
    for cam_name in Camera_list:
        instrincs = np.loadtxt(join(DATADIR, 'pose/instrincs', cam_name, '000000'))
        Instrics[cam_name] = instrincs

    # Get insrincs dic

    for fra, path in enumerate(frame_path):
        # process every frame data
        RGB_Array=[]
        xyz_Array=[]
        for camera in Camera_list:
            camera_path=join(DATADIR, 'Data', 'RGB', camera, path.replace('ply','png'))
            index_path=join(DATADIR, 'Data', 'img_index', camera, path.replace('ply','npy'))
            RGB=Image.open(camera_path)
            index=np.load(index_path)
            x_index=np.around(index[:,3])
            y_index=np.around(index[:,4])
            xy_index=(np.int32(y_index),np.int32(x_index))
            xyz_array=index[:,:3]
            RGB_array=np.array(RGB,dtype=np.uint8)[xy_index]
            xyz_Array.append(xyz_array)
            RGB_Array.append(RGB_array)
        RGB_Array=np.concatenate(RGB_Array,axis=0)
        xyz_Array=np.concatenate(xyz_Array,axis=0)

        savepath = join(RGB_Dir.replace('RGB', 'test_index'), path)

        # points_sem_p = PointCloud(array=xyz_Array, color_array=RGB_Array, frame_number=0)
        # points_sem_p.save_to_disk(savepath)
        pass

def Get_pointWithGT_from_pointWithIndex():
    DATADIR = '/media/luo/Dataset/CARLA/[divide_train][ICNET_BN] [RNNTEst]/[episode19][feature_map]'
    RGB_Dir = join(DATADIR, 'Data', 'RGB')
    Sem_Dir = join(DATADIR, 'Data', 'Semantic')
    Lidar_Dir = join(DATADIR, 'Data', 'Lidar')
    Sem_P_Dir = join(DATADIR, 'Data', 'Infer_sem')
    Camera_list = listdir(RGB_Dir)
    Lidarlist = listdir(Lidar_Dir)
    # Get each frame pcl path
    RESOLUTION = [1024, 1024]
    frame_path = listdir(join(Lidar_Dir, Lidarlist[0]))

    pointGT_dir = RGB_Dir.replace('RGB', 'points_GT')
    if not os.path.exists(pointGT_dir):
        os.makedirs(pointGT_dir)

    for fra, path in enumerate(frame_path):
        # process every frame data
        Points_GT=[]
        for camera in Camera_list:
            img_path=join(RGB_Dir,camera,path).replace('ply', 'png')
            GT_path = img_path.replace('RGB', 'Semantic')
            GT = Image.open(GT_path)
            GT = np.array(GT, dtype=np.float32)[:,:,0]

            pointsIndex_path = img_path.replace('RGB', 'img_index').replace('png', 'npy')
            pointsIndex = np.load(pointsIndex_path)
            xyz = pointsIndex[:, :3]
            xy_index = pointsIndex[:, 3:]

            points_GT = Get_around_value_of_PointWithIndex(GT, xy_index)
            points_GT=np.expand_dims(points_GT,axis=1)
            points_GT = np.concatenate([xyz, points_GT], axis=1)
            Points_GT.append(points_GT)

        Points_GT = np.concatenate(Points_GT, axis=0)
        np.save(os.path.join(pointGT_dir, path.replace('.ply', '')), np.float32(Points_GT))
        pass


            


if __name__ == '__main__':
    #CARLA_projection()
    #Test_index()
    Get_pointWithGT_from_pointWithIndex()
