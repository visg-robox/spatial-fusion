
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""DeepLab v3 models based on slim library."""

"""DeepLab v3 models based on slim library."""


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
from model_lib.ICNET_model.ICNET_model import ICNet_BN



from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np
import os

global _LOAD_PRETRAINED
_LOAD_PRETRAINED = False
_IMG_PRETRAIN = '../data_and_checkpoint/pretrain_model/deeplabv3+_apollo'


base_model = resnet_v2.resnet_v2_101
base_architecture ='resnet_v2_101'


def model_generator(num_classes, batch_norm_decay):
  def model(inputs, is_training):
    if is_training:
      output_stride = 16
    else:
      output_stride = 8

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      logits, end_points = base_model(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride,
                                      reuse = tf.AUTO_REUSE)

    inputs_size = tf.shape(inputs)[1:3]
    net = end_points[base_architecture + '/block4']
    encoder_output = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

    with tf.variable_scope("decoder",reuse = tf.AUTO_REUSE):
      with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        with arg_scope([layers.batch_norm], is_training=is_training):
          with tf.variable_scope("low_level_features"):
            low_level_features = end_points[base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
            low_level_features = layers_lib.conv2d(low_level_features, 48,
                                                   [1, 1], stride=1, scope='conv_1x1')
            low_level_features_size = tf.shape(low_level_features)[1:3]

          with tf.variable_scope("upsampling_logits"):
            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
            net = tf.concat([net, low_level_features], axis=3, name='concat')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
            
            #这里调整了feature的维度用来给后续网络作为输入从256变为128
            feature = layers_lib.conv2d(net, 128, [3, 3], stride=1, scope='conv_3x3_2')
            net = layers_lib.conv2d(feature, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')

    with tf.name_scope('load_pretrain'):
      global _LOAD_PRETRAINED
      if not _LOAD_PRETRAINED and is_training:
        _LOAD_PRETRAINED = True
        if _IMG_PRETRAIN:
          exclude = []
          exclude_name = ['global_step', 'upsampling_logits/conv_3x3_2', 'upsampling_logits/conv_1x1']
          for name in exclude_name:
            exclude_var_name = [v.name for v in tf.global_variables() if name in v.name]
            exclude += exclude_var_name
          variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
          tf.train.init_from_checkpoint(_IMG_PRETRAIN,
                                        {v.name.split(':')[0]: v for v in variables_to_restore})

    with tf.name_scope('train_var'):
      train_exclude_name = []
      train_include_name = ['upsampling_logits']

      train_exclude_var = []
      train_var = []
      for name in train_exclude_name:
        exclude_var = [v for v in tf.trainable_variables() if name in v.name]
        train_exclude_var += exclude_var
      train_var_without_exclude = [v for v in tf.trainable_variables() if v not in train_exclude_var]
      
      for name in train_include_name:
        include_var = [v for v in train_var_without_exclude if name in v.name]
        train_var += include_var
      print('train_var:', train_var)

    out = {}
    out['logits'] = logits
    out['train_var'] = train_var
    out['features'] = feature

    return out

  return model




def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
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
  with tf.variable_scope("aspp",reuse = tf.AUTO_REUSE):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net




