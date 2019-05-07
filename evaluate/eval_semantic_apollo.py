from evaluate import eval_API
import numpy as np
from PIL import Image
import os
# from scipy import misc

VOID=255
NUM_CLASS=13
CLASSNAME = [
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

VALID_CLASS=[' road ' ,
' fence ' ,
' pole ' ,
' vegatation ']


ID_COLOR = {
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
VALID_INDEX=[i for i in range(len(CLASSNAME)) if CLASSNAME[i]  in VALID_CLASS]
VOID_INDEX=[i for i in range(len(CLASSNAME)) if CLASSNAME[i] not in VALID_CLASS]


def get_file_list(Datadir):
    path_list=[]
    for i in os.listdir(Datadir):
        path_list.append(os.path.join(Datadir,i))

    return path_list


#ID to RGB
def convert_color(img):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = ID_COLOR
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(img == key)] = value
    return result

#RGB to ID
def convert_id(img):
    classes=ID_COLOR
    result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(np.all(img==value,axis=2))] = key
    return result

#Add void to Groud truth
def AddVoid(merge_img,GT_img):
    merge_void_imdex=np.where(merge_img==VOID)
    dynamic_void_index=np.where(GT_img>NUM_CLASS)

    for i in VOID_INDEX:
        index=np.where(GT_img==i)
        GT_img[index]=VOID

    GT_img[merge_void_imdex]=VOID
    GT_img[dynamic_void_index]=VOID
    void_index=np.where(GT_img==VOID)
    return GT_img,void_index


#evaluate infer, merge, and save valid pixel img for compare
def evaluate(merge_dir,infer_dir,GT_dir,RGB_dir,start_frame,end_frame,logdir,save_compare=True):
    merge_list = get_file_list(merge_dir)
    infer_list = get_file_list(infer_dir)
    GT_list = get_file_list(GT_dir)
    RGB_list=get_file_list(RGB_dir)
    merge_list.sort()
    infer_list.sort()
    GT_list.sort()
    RGB_list.sort()

    merge_total_accuracy = np.zeros([NUM_CLASS, 3], dtype=np.float32)
    infer_total_accuracy = np.zeros([NUM_CLASS, 3], dtype=np.float32)
    img_savedir = os.path.join(logdir,'Visual_Compare')
    if not os.path.isdir(img_savedir):
        os.makedirs(img_savedir)
    for i in range(start_frame,end_frame,1): #here to change frames for campare
        if i%5==0:
            print('processing',i,'/',len(merge_list))
        merge_img=np.array(Image.open(merge_list[i]),dtype=np.uint8)
        merge_img=convert_id(merge_img)
        infer_img=np.array(Image.open(infer_list[i]),dtype=np.uint8)[:,:,0]
        gt_img=np.array(Image.open(GT_list[i]),dtype=np.uint8)[:,:,0]
        gt_img-=9
        infer_img-=9
        gt_img,void_index=AddVoid(merge_img,gt_img)
        merge_total_accuracy += eval_API.getaccuracy(merge_img,gt_img,NUM_CLASS)
        infer_total_accuracy += eval_API.getaccuracy(infer_img,gt_img,NUM_CLASS)

        if save_compare:
            # def saveRGB(img,path):
            #     if not os.path.isdir(path):
            #         os.makedirs(path)
            #     img[void_index]=VOID
            #     img=convert_color(img)
            #     msic.imsave(os.path.join(path),img)
            infer_img[void_index]=VOID
            merge_img[void_index]=VOID
            infer_img=convert_color(infer_img)
            merge_img=convert_color(merge_img)
            gt_img=convert_color(gt_img)
            rgb_img=np.array(Image.open(RGB_list[i]),dtype=np.uint8)
            saveimg=np.concatenate([np.concatenate([rgb_img,gt_img],axis=0),np.concatenate([infer_img,merge_img],axis=0)],axis=1)

            # misc.imsave(os.path.join(img_savedir,str(i)+'.png'),saveimg)



    eval_API.eval_print_save(merge_total_accuracy[VALID_INDEX], 'merge', logdir,
                             np.array(CLASSNAME)[VALID_INDEX])
    eval_API.eval_print_save(infer_total_accuracy[VALID_INDEX], 'infer', logdir,
                             np.array(CLASSNAME)[VALID_INDEX])




# class Eval_Infer(object):
#     def __init__(self,infer_img_list,label_img_list,logdir,**kwargs):
#         self.class_name=CLASSNAME
#         self.total_accuracy = np.zeros([len(self.class_name), 3], dtype=np.float32)
#         self.infer_img_list=infer_img_list
#         self.merge_img_list=merge_img_list
#         self.label_img_list=label_img_list
#         self.logdir=logdir
#         self.method_name='CRALAF0'
#         self.ignore=[]
#         self.set(**kwargs)

#     def set(self, **kwargs):
#         for key, value in kwargs.items():
#             if not hasattr(self, key):
#                 raise ValueError('sensor.Sensor: no key named %r' % key)
#             setattr(self, key, value)

#     def calculate(self):
#         for i,path in enumerate(self.infer_img_list):
#             infer_img=Image.open(path)
#             infer_img=np.array(infer_img)[:,:,0]
#             label_img = Image.open(self.label_img_list[i])
#             label_img = np.array(label_img)[:, :, 0]
#             self.total_accuracy+=eval_API.getaccuracy(infer_img,label_img,len(self.class_name))

#     def eval_print(self):
#         valid_index=[i for i in range(len(self.class_name)) if self.class_name[i] not in self.ignore]
#         valid_index=np.array(valid_index,dtype=np.uint16)


#         eval_API. eval_print_save(self.total_accuracy[valid_index],self.method_name,self.logdir,classname=np.array(self.class_name)[valid_index])





if __name__ == '__main__':
    Logdir='/media/luo/temp/apollo/My_pcl/ver_deeplab/episode5_Record005/Camera 5/eval/test'
    Infer_Dir='/media/luo/temp/apollo/My_pcl/ver_deeplab/episode5_Record005/Camera 5/Pictures/Infer/ID'
    Merge_Dir='/media/luo/temp/apollo/My_pcl/ver_deeplab/episode5_Record005/Camera 5/Pictures/Merge'
    RGB_Dir='/media/luo/temp/apollo/My_pcl/ver_deeplab/episode5_Record005/Camera 5/Pictures/RGB'
    GT_Dir='/media/luo/temp/apollo/My_pcl/ver_deeplab/episode5_Record005/Camera 5/Pictures/GT/ID'
    Start_Frame=20
    End_Frame=40
    evaluate(Merge_Dir,Infer_Dir,GT_Dir,RGB_Dir,Start_Frame,End_Frame,Logdir,save_compare=True)










