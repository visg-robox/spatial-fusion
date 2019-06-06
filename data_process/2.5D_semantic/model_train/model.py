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
from model_lib.ICNET_model.ICNET_model import ICNet_BN



from tensorflow.contrib.slim.nets import resnet_v2
import numpy as np
import os

global _LOAD_PRETRAINED
_LOAD_PRETRAINED = False
_IMG_PRETRAIN = '../data_and_checkpoint/pretrain_model/ICNET_cityscape_pretrain'

#special parameter

def model_generator(num_classes, batch_norm_decay):
    def VGG_model(input, is_training):
        image = input
        #convert to bgr to restore weight
        bgr = tf.split(image, num_or_size_splits=3, axis=-1)
        image = tf.concat([bgr[2], bgr[1], bgr[0]], axis=-1)
        inputs_size = tf.shape(image)[1:3]

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):

            net = ICNet_BN({'data': image}, is_training=is_training, num_classes=num_classes,
                           filter_scale=1)
            features = net.layers['sub12_sum_interp']
            logits = net.layers['conv6_cls']

        with tf.name_scope('load_pretrain'):
            global _LOAD_PRETRAINED
            if not _LOAD_PRETRAINED:
                _LOAD_PRETRAINED = True
                if _IMG_PRETRAIN:
                    exclude = []
                    exclude_name = ['global_step' ,'conv6_cls']
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
        out['logits'] = logits
        out['train_var'] = train_var
        out['features'] = features

        return out

    return VGG_model









