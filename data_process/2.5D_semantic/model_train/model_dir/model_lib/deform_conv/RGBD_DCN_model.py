"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets.resnet_utils import conv2d_same
from tensorflow.python.ops import init_ops
from cnn_model.Deformable_layer import deform_conv,Get_project_deform_offset_v2,Get_transform_deform_offset

from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np
import os

global _LOAD_PRETRAINED
_LOAD_PRETRAINED = False


_DEPTH_PRETRAIN = ''
#_IMG_PRETRAIN = 'cityscape_model/resnet/101_dense_nodecode'
_IMG_PRETRAIN = 'cityscape_model/pretrain_model/resnet_v2_101_2017_04_14'
_TRAIN_OFFSET= True
_OUTPUT_STRIDE = 16
_DILATE_RATE = [4, 8, 12]
_OFFSET_LR = [0.3, 0.6, 0.9]
MODEL_DIR='cityscape_model/resnet/multi-gpu'
_img_base_model = resnet_v2.resnet_v2_101
_img_scope = 'resnet_v2_101'
_MESSAGE = 'res 101, dilate rate 4,8,12 train from imgnet with only scale shift with lr [0.3, 0.6, 0.9] as for one pixel step'

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)
with open(MODEL_DIR+'/README.txt','w+') as log:
    log.write(_MESSAGE)

def dense_deform_module(inputs, output_stride, is_training, batch_norm_decay, depth=64, offset = None, transition_depth = 1024):
    if output_stride not in [8, 16, 32]:
        raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates =_DILATE_RATE #same ERF as ASPP
    atrous_rates = [int(rate*(16/output_stride)) for rate in atrous_rates]
    shape = tf.shape(inputs)[0:3]
    #define the transform
    project_shape =  tf.concat([shape, tf.constant([2, 2], dtype=shape.dtype)], axis=0)
    #constaint with mask
    '''
    [y,x] dot 1 0 x k = offset
              0 1 
    
    '''

    offset_lr = _OFFSET_LR
    mask_1 = tf.multiply(tf.constant([1,0,0,1],dtype=tf.float32,shape=[2,2]),tf.constant(offset_lr[0]))
    mask_2 = tf.multiply(tf.constant([1,0,0,1],dtype=tf.float32,shape=[2,2]),tf.constant(offset_lr[1]))
    mask_3 = tf.multiply(tf.constant([1,0,0,1],dtype=tf.float32,shape=[2,2]),tf.constant(offset_lr[2]))


    transition = layers_lib.conv2d(inputs, transition_depth, [1, 1], stride=1, scope="conv_1x1_transition")
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        with arg_scope([layers.batch_norm], is_training=is_training):
            feature_1 = transition
            conv_1x1_1 = layers_lib.conv2d(feature_1, depth, [1, 1], stride=1, scope="conv_1x1_1")

            offset_1 = layers_lib.conv2d(feature_1, 4,
                                              [2, 2], activation_fn=None,
                                            weights_initializer = init_ops.zeros_initializer,
                                            biases_initializer = init_ops.zeros_initializer,
                                            normalizer_fn=None, scope='conv_offset_1')

            offset_1 = tf.reshape(offset_1, project_shape)
            offset_1 = tf.multiply(offset_1, mask_1)
            offset_1= Get_project_deform_offset_v2(tranform_mat=offset_1, dilated_rate=1)
            if offset:
                offset_1 = offset[0]
            dcn_3x3_1 = deform_conv(conv_1x1_1, offset_1, k_w=3, k_h=3, c_o =depth,
                                    s_w=1,
                                    s_h=1,
                                    num_deform_group=1, biased=False, name='conv1', rate=atrous_rates[0],
                                    train_offset=_TRAIN_OFFSET)

            feature_2 = tf.concat([feature_1, dcn_3x3_1], axis=3)
            conv_1x1_2 = layers_lib.conv2d(feature_2, depth, [1, 1], stride=1, scope="conv_1x1_2")
            offset_2 = layers_lib.conv2d(feature_2, 4,
                                         [2, 2], activation_fn=None,
                                         weights_initializer=init_ops.zeros_initializer,
                                         biases_initializer=init_ops.zeros_initializer,
                                         normalizer_fn=None, scope='conv_offset_2')
            offset_2 = tf.reshape(offset_2, project_shape)
            offset_2 = tf.multiply(offset_2, mask_2)
            offset_2 = Get_project_deform_offset_v2(tranform_mat=offset_2,dilated_rate=1)
            if offset:
                offset_2 = offset[1]
            dcn_3x3_2= deform_conv(conv_1x1_2, offset_2, k_w=3, k_h=3, c_o=depth,
                                    s_w=1,
                                    s_h=1,
                                    num_deform_group=1, biased=False, name='conv2', rate=atrous_rates[1],
                                    train_offset=_TRAIN_OFFSET)

            feature_3 = tf.concat([feature_2, dcn_3x3_2], axis=3)
            conv_1x1_3 = layers_lib.conv2d(feature_3, depth, [1, 1], stride=1, scope="conv_1x1_3")
            offset_3 = layers_lib.conv2d(feature_3, 4,
                                         [2, 2], activation_fn=None,
                                         weights_initializer=init_ops.zeros_initializer,
                                         biases_initializer=init_ops.zeros_initializer,
                                         normalizer_fn=None, scope='conv_offset_3')
            offset_3 = tf.reshape(offset_3, project_shape)
            offset_3 = tf.multiply(offset_3, mask_3)
            offset_3 = Get_project_deform_offset_v2(tranform_mat=offset_3, dilated_rate=1)
            if offset:
                offset_3 = offset[2]
            dcn_3x3_3 = deform_conv(conv_1x1_3, offset_3, k_w=3, k_h=3,
                                    c_o=depth,
                                    s_w=1,
                                    s_h=1, num_deform_group=1, biased=False, name='conv3', rate=atrous_rates[2],
                                    train_offset=_TRAIN_OFFSET)

            # feature_4 = tf.concat([feature_3, dcn_3x3_3], axis=3)
            # conv_1x1_4 = layers_lib.conv2d(feature_4, depth*2, [1, 1], stride=1, scope="conv_1x1_4")
            # offset_4 = layers_lib.conv2d(feature_4, 18, [1, 1],
            #                                 activation_fn=None,
            #                                 weights_initializer=init_ops.zeros_initializer(),
            #                                 normalizer_fn=None, scope='conv_offset_4')
            # if offset:
            #     offset_4 = offset[3]
            # dcn_3x3_4 = deform_conv(conv_1x1_4, offset_4, k_w=3, k_h=3,
            #                         c_o=depth,
            #                         s_w=1,
            #                         s_h=1, num_deform_group=1, biased=False, name='conv4', rate=atrous_rates[3],
            #                         train_offset=_TRAIN_OFFSET)

            # inputs_size = inputs.get_shape().as_list()[1:3]
            # image_level_features = tf.reduce_mean(dcn_3x3_3, [1, 2], name='global_average_pooling', keep_dims=True)
            # # 1x1 convolution with 256 filters( and batch normalization)
            #
            # image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

            out = tf.concat([feature_3, dcn_3x3_3], axis=3)
            out = layers_lib.conv2d(out, depth, [1, 1], stride=1, scope="conv_1x1_out")
            offset_ret = tf.concat(
                [tf.expand_dims(offset_3, axis=0),
                    tf.expand_dims(offset_2, axis=0), tf.expand_dims(offset_1, axis=0)],
                axis=0)
            outdict={}
            outdict['net'] = out
            outdict['offset'] = offset_ret
            return outdict


def dense_deform_modele_decode(inputs, is_training, batch_norm_decay, depth=64, offset = None):
    with tf.variable_scope('dense_deform_decode'):
        atrous_rates = [1, 2]
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            with arg_scope([layers.batch_norm], is_training=is_training):
                conv_1x1_1 = layers_lib.conv2d(inputs, depth*2, [1, 1], stride=1, scope="conv_1x1_1")
                offset_1 = layers_lib.conv2d(conv_1x1_1, 18, [3, 3],
                                                     activation_fn=None,
                                                     weights_initializer=init_ops.zeros_initializer(),
                                                     normalizer_fn=None, scope='conv_offset_1')
                if offset:
                    offset_1 = offset[0]
                dcn_3x3_1 = deform_conv(conv_1x1_1, offset_1, k_w=3, k_h=3, c_o =depth,
                                        s_w=1,
                                        s_h=1,
                                        num_deform_group=1, biased=False, name='conv1', rate=atrous_rates[0],
                                        train_offset=_TRAIN_OFFSET)

                feature_2 = tf.concat([inputs, dcn_3x3_1], axis=3)
                conv_1x1_2 = layers_lib.conv2d(feature_2, depth*2, [3, 3], stride=1, scope="conv_1x1_2")
                offset_2 = layers_lib.conv2d(conv_1x1_2, 18, [3, 3],
                                                     activation_fn=None,
                                                     weights_initializer=init_ops.zeros_initializer(),
                                                     normalizer_fn=None, scope='conv_offset_2')
                if offset:
                    offset_2 = offset[1]
                dcn_3x3_2= deform_conv(conv_1x1_2, offset_2, k_w=3, k_h=3, c_o=depth,
                                        s_w=1,
                                        s_h=1,
                                        num_deform_group=1, biased=False, name='conv2', rate=atrous_rates[1],
                                        train_offset=_TRAIN_OFFSET)

                out = tf.concat([inputs, dcn_3x3_2])
                return out





def atrous_spatial_pyramid_pooling(inputs, output_stride, is_training, weight_decay,
                                   batch_norm_decay, depth=256):
    """Atrous Spatial Pyramid Pooling.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      is_training: A boolean denoting whether the input is for training.
      depth: The depth of the ResNet unit output.

    Returns:
      The atrous spatial pyramid pooling output.
    """
    with tf.variable_scope("aspp"):
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')

        atrous_rates = [4, 8, 12]
        if output_stride == 8:
            atrous_rates = [2 * rate for rate in atrous_rates]


        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0],
                                       scope='conv_3x3_1')
        conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1],
                                       scope='conv_3x3_2')
        conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2],
                                       scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
            # global average pooling
            image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
            # 1x1 convolution with 256 filters( and batch normalization)
            image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1,
                                                     scope='conv_1x1')
            # bilinearly upsample features
            image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        temp = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3,
                        name='concat')
        temp = layers_lib.conv2d(temp, depth, [1, 1], stride=1, scope='conv_1x1_concat')
        return temp


def rgbd_dcn_generator(num_classes, output_stride, batch_norm_decay):
    def VGG_model(input, is_training):
        image = input
        inputs_size = tf.shape(image)[1:3]

        # some op parameter with arg_scope
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            image_input = image
            img_net, img_end_points = _img_base_model(image_input,
                                                        num_classes=None,
                                                        reuse= tf.AUTO_REUSE,
                                                        is_training=is_training,
                                                        global_pool=False,
                                                        output_stride=_OUTPUT_STRIDE)
        with tf.variable_scope('dense_deform',reuse=tf.AUTO_REUSE):
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    img_out = img_end_points[_img_scope + '/block4']
                    dense_out = dense_deform_module(img_out, output_stride=_OUTPUT_STRIDE, batch_norm_decay = batch_norm_decay, is_training= is_training, depth = 256, transition_depth=1024)
                    high_feature = dense_out['net']
                    offset = dense_out['offset']

        with tf.variable_scope('decode',reuse=tf.AUTO_REUSE):
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                with arg_scope([layers.batch_norm], is_training=is_training):

                    logits_high = layers_lib.conv2d(high_feature, num_classes, [1, 1], activation_fn=None,
                                                normalizer_fn=None, scope='conv_1x1_predict')
                    logits_high = tf.image.resize_bilinear(logits_high, inputs_size, name='upsample_3')


        with tf.name_scope('load_pretrain'):
            global _LOAD_PRETRAINED
            if not _LOAD_PRETRAINED:
                _LOAD_PRETRAINED = True
                if _IMG_PRETRAIN:
                    exclude = []
                    exclude_name = ['global_step','dense_deform','decode']
                    for name in exclude_name:
                        exclude_var_name = [v.name for v in tf.global_variables() if name in v.name]
                        exclude += exclude_var_name
                    variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
                    tf.train.init_from_checkpoint(_IMG_PRETRAIN,
                                                {v.name.split(':')[0]: v for v in variables_to_restore})

        with tf.name_scope('train_var'):
            train_exclude_name = []
            train_exclude_var = []
            for name in train_exclude_name:
                exclude_var= [v for v in tf.trainable_variables() if name in v.name]
                train_exclude_var += exclude_var
            train_var = [v for v in tf.trainable_variables() if v not in train_exclude_var]
            print(train_var)


        out = {}
        out['logits'] = logits_high
        offset = tf.transpose(offset, [0, 1, 4, 2, 3])
        out['offset'] = offset
        out['train_var'] = train_var

        return out

    return VGG_model









