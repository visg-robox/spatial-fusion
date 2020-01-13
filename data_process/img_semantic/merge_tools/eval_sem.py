import eval_API
import numpy as np
from PIL import Image
import os


CLASSNAME = ['void', 'Buildings', 'Fences', 'Other', 'Pedestrians', 'Poles', 'RoadLines', 'Roads', 'Sidewalks',
             'Vegetation', 'Vehicles', 'Walls', 'TrafficSigns']
IGNORE=['void','Pedestrians','Vehicles','Vegetation', 'TrafficSigns']

class Eval_Infer(object):
    def __init__(self,infer_img_list,label_img_list,logdir,**kwargs):
        self.class_name=CLASSNAME
        self.total_accuracy = np.zeros([len(self.class_name), 3], dtype=np.float32)
        self.infer_img_list=infer_img_list
        self.label_img_list=label_img_list
        self.logdir=logdir
        self.method_name='CRALAF0'
        self.ignore=[]
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError('sensor.Sensor: no key named %r' % key)
            setattr(self, key, value)

    def calculate(self):
        for i,path in enumerate(self.infer_img_list):
            infer_img=Image.open(path)
            infer_img=np.array(infer_img)[:,:,0]
            label_img = Image.open(self.label_img_list[i])
            label_img = np.array(label_img)[:, :, 0]
            self.total_accuracy+=eval_API.getaccuracy(infer_img,label_img,len(self.class_name))

    def eval_print(self):
        valid_index=[i for i in range(len(self.class_name)) if self.class_name[i] not in self.ignore]
        valid_index=np.array(valid_index,dtype=np.uint16)


        eval_API. eval_print_save(self.total_accuracy[valid_index],self.method_name,self.logdir,classname=np.array(self.class_name)[valid_index])

def get_file_list(Datadir):
    path_list=[]
    for i in os.walk(Datadir):
        if i[1]==[]:
            for p in i[2]:
                path_list.append(os.path.join(i[0],p))
    return path_list

if __name__ == '__main__':
    Logdir='/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/out_lidar_camera_semantic/episode_0000/Eval/v1_F0/result'
    Merge_dir = '/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/out_lidar_camera_semantic/episode_0000/Eval/v1_F0/merge'
    merge_path = get_file_list(Merge_dir)
    merge_path = [i for i in merge_path if 'color' not in i]
    merge_path.sort()
    merge_path=merge_path[20:60]

    infer_path = [i.replace('merge', 'infer') for i in merge_path]
    label_path = [i.replace('merge', 'label') for i in merge_path]
    print(label_path)

    ICNET_eval=Eval_Infer(infer_img_list=infer_path,label_img_list=label_path,logdir=Logdir,ignore=IGNORE,method_name='ICNET')
    ICNET_eval.calculate()
    ICNET_eval.eval_print()

    merge_eval=Eval_Infer(infer_img_list=merge_path,label_img_list=label_path,logdir=Logdir,ignore=IGNORE,method_name='Merge')
    merge_eval.calculate()
    merge_eval.eval_print()

    merge_path = [i for i in merge_path if 'color'  in i]
    merge_path = merge_path[0:40]

    infer_path = [i.replace('merge', 'infer') for i in merge_path]
    label_path = [i.replace('merge', 'label') for i in merge_path]





