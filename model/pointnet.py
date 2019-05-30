import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


#????? wait for correct
input_dim = 3

num_point = 2000
dim = 0

class Pointnet(nn.Module):
    def __init__(self, feature_dim = 64):
        super(Pointnet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, feature_dim, (1,3))
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, 1)

        self.conv3 = nn.Conv2d(, feature_dim, 1)
        self.conv4 = nn.Conv2d(feature_dim, feature_dim*2, 1)
        self.conv5 = nn.Conv2d(feature_dim*2, feature_dim*16, 1)

        self.conv3_1 = nn.Conv2d(dim, feature_dim * 8, 1)
        self.conv3_2 = nn.Conv2d(feature_dim*8, feature_dim * 4, 1)
        self.conv3_3 = nn.Conv2d(feature_dim*4, feature_dim * 2, 1)
        self.conv3_4 = nn.Conv2d(feature_dim * 2, feature_dim * 2, 1)
        self.conv3_5 = nn.Conv2d(feature_dim * 2, 50, 1)

        self.bn1 = nn.BatchNorm2d(feature_dim)
        self.bn2 = nn.BatchNorm2d(feature_dim)
        self.bn3 = nn.BatchNorm2d(feature_dim)
        self.bn4 = nn.BatchNorm2d(feature_dim * 2)
        self.bn5 = nn.BatchNorm2d(feature_dim * 16)

        self.bn3_1 = nn.BatchNorm2d(feature_dim*8)
        self.bn3_2 = nn.BatchNorm2d(feature_dim*4)
        self.bn3_3 = nn.BatchNorm2d(feature_dim*2)
        self.bn3_4 = nn.BatchNorm2d(feature_dim*2)

        self.max_pool = nn.MaxPool2d([num_point, 1])


    def forward(self, point_cloud):

        """ Classification PointNet, input is BxNx3, output BxNx50 """
        # point_cloud [bz, num_cloud]
        batch_size = point_cloud.shape[0]
        num_point = point_cloud.shape[1]
        end_points = {}

        transform = self.input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = point_cloud.mul(transform)
        input_image = point_cloud_transformed.unsqeeze(-1)

        net = self.conv1(input_image)
        net = self.bn1(net)
        net = self.conv2(net)
        net = self.net2(net)

        transform = self.feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform

        net_transformed = net.squeeze(2).mul(transform)
        point_feat = net_transformed.unsqueeze(2)
        print(point_feat)

        net = self.conv3(point_feat)
        net = self.bn3(net)
        net = self.conv4(net)
        net = self.bn4(net)
        net = self.conv5(net)
        net = self.bn(net)

        global_feat = self.max_pool(net)
        print(global_feat)

        global_feat_expand = global_feat.expand(-1, num_point, -1, -1)
        concat_feat = torch.cat((point_feat, global_feat_expand), 3)
        print(concat_feat)

        net = self.conv3_1(concat_feat)
        net = self.bn3_1(net)
        net = self.conv3_2(net)
        net = self.bn3_2(net)
        net = self.conv3_3(net)
        net = self.bn3_3(net)
        net = self.conv3_4(net)
        net = self.bn3_4(net)
        net = self.conv3_5(net)
        net = net.squeeze(2)  # BxNxC

        return net, end_points

    def input_transform_net(self, point_cloud, is_training, bn_decay=None, K=3):
        """ Input (XYZ) Transform Net, input is BxNx3 gray image
            Return:
                Transformation matrix of size 3xK """
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value

        input_image = tf.expand_dims(point_cloud, -1)
        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='tconv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='tconv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='tconv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='tmaxpool')

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='tfc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='tfc2', bn_decay=bn_decay)

        with tf.variable_scope('transform_XYZ') as sc:
            assert (K == 3)
            weights = tf.get_variable('weights', [256, 3 * K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [3 * K],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, 3, K])
        return transform

    def feature_transform_net(self, inputs, is_training, bn_decay=None, K=64):
        """ Feature Transform Net, input is BxNx1xK
            Return:
                Transformation matrix of size KxK """
        batch_size = inputs.get_shape()[0].value
        num_point = inputs.get_shape()[1].value

        net = tf_util.conv2d(inputs, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='tconv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='tconv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='tconv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='tmaxpool')

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='tfc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='tfc2', bn_decay=bn_decay)

        with tf.variable_scope('transform_feat') as sc:
            weights = tf.get_variable('weights', [256, K * K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [K * K],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, K, K])
        return transform