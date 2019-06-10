import os
import numpy as np
import tensorflow as tf

def get_filename(dir):
    file_name=[]
    namelist=os.listdir(dir)
    namelist.sort()
    for name in namelist:
        file_name.append(dir+'/'+name)
    return file_name

def image_mirroring(img, label):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    
    return img, label

def image_scaling(img, label):
    scale = tf.random_uniform([1], minval=0.7, maxval=1.3, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, label
    
def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop

def read_labeled_image_list(data_dir):
    imgl_names=get_filename(data_dir+'/image')
    labels_names=get_filename(data_dir+'/label')
    
    return imgl_names,labels_names
    
    
   


def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label, img_mean): # optional pre-processing arguments
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    img = tf.image.decode_image(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label = tf.image.decode_image(label_contents, channels=1)
    h, w = input_size
    img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    if input_size is not None:
       
        if random_scale:
            img, label = image_scaling(img, label)

        if random_mirror:
            img, label = image_mirroring(img, label)

        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return img, label

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, input_size,
                  random_scale, random_mirror, ignore_label, img_mean, coord):

        self.data_dir = data_dir
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror, ignore_label, img_mean)

    def dequeue(self, num_elements):
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
        return image_batch, label_batch


def datafromtfrecord(recordname,shape,imgmean,batchsize,capacity,min):

    height,width=shape
    print(height,width)

    reader = tf.TFRecordReader()

    print('read ' + recordname + '\n')

    recordtensor = tf.train.string_input_producer([recordname])
    _, serialized_example = reader.read(recordtensor)


    features = tf.parse_single_example(serialized_example, features={
        'label_raw': tf.FixedLenFeature([], tf.string),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    image=tf.cast(image,tf.float32)


    image.set_shape([height * width * 3])
    image = tf.reshape(image, [height, width, 3])
    image-=imgmean

    label.set_shape([height* width * 1])
    label = tf.reshape(label, [height, width, 1])

    image, label = tf.train.shuffle_batch([image, label],
                                  batch_size=batchsize,
                                  capacity=capacity,
                                  min_after_dequeue=min)
    return image,label