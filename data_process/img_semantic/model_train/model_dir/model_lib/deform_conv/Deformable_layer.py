import tensorflow as tf
from cnn_model.deform_conv_layer import deform_conv_op
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops


DEFAULT_PADDING='SAME'
import numpy as np


def validate_padding(padding):
    assert padding in ('SAME', 'VALID')

def make_var(name, shape, initializer='zeros', trainable=True, regularizer=None):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)


def deform_conv( data, offset, k_h, k_w, c_o, s_h, s_w, num_deform_group, name, num_groups = 1, rate = 1, biased=True, relu=True,
                padding=DEFAULT_PADDING, trainable=True, initializer=None, train_offset=False):
    """ contribution by miraclebiu, and biased option"""

    validate_padding(padding)
    c_i = data.get_shape()[-1]
    trans2NCHW = lambda x:tf.transpose(x, [0, 3 ,1 ,2])
    trans2NHWC = lambda x:tf.transpose(x, [0, 2 ,3, 1])
    # deform conv only supports NCHW

    with tf.variable_scope(name) as scope:
        if not train_offset:
            offset_shape = tf.shape(data)
            offset_shape = tf.slice(offset_shape, [0], [3])
            offset_shape = tf.concat([offset_shape, tf.constant([k_h * k_w * 2], dtype=tf.int32)], axis=0)
            offset = tf.zeros(offset_shape, dtype=tf.float32)


        data = trans2NCHW(data)
        offset = trans2NCHW(offset)
        dconvolve = lambda i, k, o: deform_conv_op.deform_conv_op(
            i, k, o, strides=[1, 1, s_h, s_w], rates=[1, 1, rate, rate], padding=padding, num_groups=num_groups,
            deformable_group=num_deform_group)

        # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
        init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(
            factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        kernel = make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable)
        kernel = tf.transpose(kernel, [3, 2, 0, 1])
        print(data, kernel, offset)
        dconv = trans2NHWC(dconvolve(data, kernel, offset))
        if biased:
            biases = make_var('biases', [c_o], init_biases, trainable)
            if relu:
                bias = tf.nn.bias_add(dconv, biases)
                return tf.nn.relu(bias)
            return tf.nn.bias_add(dconv, biases)
        else:
            if relu:
                dconv=layers.batch_norm(dconv, activation_fn=nn_ops.relu)
            return dconv

def Get_project_deform_offset(tranform_mat, k_h = 3, k_w = 3, dilated_rate =1):
    input_shape = tf.shape(tranform_mat)[0:3]
    x_field = (k_w-1)//2*dilated_rate
    x_shift = tf.constant(np.arange(-x_field,x_field+1,dilated_rate),dtype= tf.float32,shape=[1,k_w,1])
    x_mat = tf.tile(x_shift, (k_h, 1, 1))
    y_field = (k_h - 1) // 2 * dilated_rate
    y_shift =  tf.constant(np.arange(-y_field,y_field+1,dilated_rate),dtype= tf.float32,shape=[k_h,1,1])
    y_mat = tf.tile(y_shift, (1, k_w, 1))
    xy_mat = tf.concat([y_mat,x_mat], axis=2)
    xy_mat = tf.reshape(xy_mat,[1, 1, 1, -1, 2])
    xy_mat = tf.tile(xy_mat,(tf.concat([input_shape,tf.constant([1,1],dtype=input_shape.dtype)],axis=0)))
    offset = tf.matmul(xy_mat, tranform_mat) - xy_mat
    offset = tf.reshape(offset, tf.concat([input_shape,tf.constant([k_w * k_h * 2],dtype=input_shape.dtype)],axis=0))
    return offset

def Get_project_deform_offset_v2(tranform_mat, k_h = 3, k_w = 3, dilated_rate =1):
    input_shape = tf.shape(tranform_mat)[0:3]
    x_field = (k_w-1)//2*dilated_rate
    x_shift = tf.constant(np.arange(-x_field,x_field+1,dilated_rate),dtype= tf.float32,shape=[1,k_w,1])
    x_mat = tf.tile(x_shift, (k_h, 1, 1))
    y_field = (k_h - 1) // 2 * dilated_rate
    y_shift =  tf.constant(np.arange(-y_field,y_field+1,dilated_rate),dtype= tf.float32,shape=[k_h,1,1])
    y_mat = tf.tile(y_shift, (1, k_w, 1))
    xy_mat = tf.concat([y_mat,x_mat], axis=2)
    xy_mat = tf.reshape(xy_mat,[1, 1, 1, -1, 2])
    xy_mat = tf.tile(xy_mat,(tf.concat([input_shape,tf.constant([1,1],dtype=input_shape.dtype)],axis=0)))
    offset = tf.matmul(xy_mat, tranform_mat)
    offset = tf.reshape(offset, tf.concat([input_shape,tf.constant([k_w * k_h * 2],dtype=input_shape.dtype)],axis=0))
    return offset

def Get_transform_deform_offset(tranform_mat, k_h = 3, k_w = 3, dilated_rate =1):
    '''
    This fuction apply project-like deform on origin dilated grid just to simulate the real world surface empirical transform and return offset by substract origin grid coordinate.
    The dilated grid is centered around zero. And the operation is tf.matmul(kernel size**2 x [y x 1], [3x3]) with homogeneous projective coordinate system.



    :param tranform_mat:
    :param k_h:
    :param k_w:
    :param dilated_rate:
    :return:
    '''
    input_shape = tf.shape(tranform_mat)[0:3]
    x_field = (k_w-1)//2*dilated_rate
    x_shift = tf.constant(np.arange(-x_field,x_field+1,dilated_rate),dtype= tf.float32,shape=[1,k_w,1])
    x_mat = tf.tile(x_shift, (k_h, 1, 1))
    y_field = (k_h - 1) // 2 * dilated_rate
    y_shift =  tf.constant(np.arange(-y_field,y_field+1,dilated_rate),dtype= tf.float32,shape=[k_h,1,1])
    y_mat = tf.tile(y_shift, (1, k_w, 1))
    ones = tf.ones_like(y_mat,dtype=tf.float32)
    xy_mat = tf.concat([y_mat,x_mat,ones], axis=2)
    xy_mat = tf.reshape(xy_mat,[1, 1, 1, -1, 3])
    xy_mat = tf.tile(xy_mat,(tf.concat([input_shape,tf.constant([1,1],dtype=input_shape.dtype)],axis=0)))
    offset = tf.matmul(xy_mat, tranform_mat)
    xy = tf.slice(offset, (0,0,0,0,0), tf.concat([tf.shape(xy_mat)[0:4],tf.constant([2],dtype=input_shape.dtype)],axis=0))
    z = tf.slice(offset, (0,0,0,0,2), tf.concat([tf.shape(xy_mat)[0:4],tf.constant([1],dtype=input_shape.dtype)],axis=0))
    xy = tf.multiply(xy,z)
    offset = xy - tf.slice(xy_mat,(0,0,0,0,0), tf.concat([tf.shape(xy_mat)[0:4],tf.constant([2],dtype=input_shape.dtype)],axis=0))
    offset = tf.reshape(offset, tf.concat([input_shape,tf.constant([k_w * k_h * 2],dtype=input_shape.dtype)],axis=0))
    return offset


def Get_two_level_affine_deform_offset(tranform_mat, k_h = 3, k_w = 3, dilated_rate =1):
    input_shape = tf.shape(tranform_mat)[0:3]

    x_field = (k_w-1)//2*dilated_rate
    x_shift = tf.constant(np.arange(-x_field,x_field+1,dilated_rate),dtype= tf.float32,shape=[1,k_w,1])
    x_squre_shift = tf.div(tf.square(x_shift),tf.constant(dilated_rate,dtype=tf.float32))
    x_mat = tf.tile(x_shift, (k_h, 1, 1))
    x_squre_mat = tf.tile(x_squre_shift, (k_h, 1, 1))

    y_field = (k_h - 1) // 2 * dilated_rate
    y_shift =  tf.constant(np.arange(-y_field,y_field+1,dilated_rate),dtype= tf.float32,shape=[k_h,1,1])
    y_squre_shift = tf.div(tf.square(y_shift),tf.constant(dilated_rate,dtype=tf.float32))
    y_mat = tf.tile(y_shift, (1, k_w, 1))
    y_squre_mat = tf.tile(y_squre_shift,(1, k_w, 1))

    index_mat = tf.concat([y_mat, x_mat], axis=2)
    index_mat = tf.reshape(index_mat, [1, 1, 1, -1, 2])
    index_mat = tf.tile(index_mat, (tf.concat([input_shape, tf.constant([1, 1], dtype=input_shape.dtype)], axis=0)))

    xy_mat = tf.concat([y_mat, x_mat, y_squre_mat, x_squre_mat], axis=2)
    xy_mat = tf.reshape(xy_mat,[1, 1, 1, -1, 4])
    xy_mat = tf.tile(xy_mat,(tf.concat([input_shape,tf.constant([1,1],dtype=input_shape.dtype)],axis=0)))

    offset = tf.matmul(xy_mat, tranform_mat) - index_mat
    offset = tf.reshape(offset, tf.concat([input_shape,tf.constant([k_w * k_h * 2],dtype=input_shape.dtype)],axis=0))
    return offset


def deform_conv_back(data, offset, k_h, k_w, c_o, s_h, s_w, num_deform_group, name, num_groups=1, rate=1, biased=True,
                relu=True,
                padding=DEFAULT_PADDING, trainable=True, initializer=None):
    """ contribution by miraclebiu, and biased option"""
    validate_padding(padding)
    c_i = data.get_shape()[-1]
    trans2NCHW = lambda x: tf.transpose(x, [0, 3, 1, 2])
    trans2NHWC = lambda x: tf.transpose(x, [0, 2, 3, 1])
    # deform conv only supports NCHW
    data = trans2NCHW(data)
    offset = trans2NCHW(offset)
    dconvolve = lambda i, k, o: deform_conv_op.deform_conv_op(
        i, k, o, strides=[1, 1, s_h, s_w], rates=[1, 1, rate, rate], padding=padding, num_groups=num_groups,
        deformable_group=num_deform_group)
    with tf.variable_scope(name) as scope:

        # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
        init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(
            factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        kernel = make_var('weights', [c_o, c_i, k_h, k_w], init_weights, trainable)
        print(data, kernel, offset)
        dconv = trans2NHWC(dconvolve(data, kernel, offset))
        if biased:
            biases = make_var('biases', [c_o], init_biases, trainable)
            if relu:
                bias = tf.nn.bias_add(dconv, biases)
                return tf.nn.relu(bias)
            return tf.nn.bias_add(dconv, biases)
        else:
            if relu:
                return tf.nn.relu(dconv)
            return dconv



