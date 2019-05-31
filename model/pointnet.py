import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import common

###########pointnet####################################
#   This is pointnet pytorch implementation for our SPNET,
#   The input should be [B, C_i, N, 1]
#   The output is the geometric feature of each voxel in a block is [B, C_o, N, 1]


NUM_POINT = 2000
CAPACITY = 64
INPUT_FEATURE_DIM = 3

class Pointnet(nn.Module):
    def __init__(self, capacity = CAPACITY, class_num = common.class_num, input_dim = INPUT_FEATURE_DIM):
        super(Pointnet, self).__init__()
        self.input_transform = input_transform_net(capacity = capacity, k = input_dim)
        self.conv1 = nn.Conv2d(input_dim, capacity, 1)
        self.conv2 = nn.Conv2d(capacity, capacity, 1)

        self.conv3 = nn.Conv2d(capacity, capacity, 1)
        self.conv4 = nn.Conv2d(capacity, capacity*2, 1)
        self.conv5 = nn.Conv2d(capacity*2, capacity*16, 1)

        self.conv3_1 = nn.Conv2d(capacity *9, capacity * 8, 1)
        self.conv3_2 = nn.Conv2d(capacity*8, capacity * 4, 1)
        self.conv3_3 = nn.Conv2d(capacity*4, capacity * 2, 1)
        self.conv3_4 = nn.Conv2d(capacity * 2, capacity * 2, 1)
        self.conv3_5 = nn.Conv2d(capacity * 2, class_num, 1)

        self.bn1 = nn.BatchNorm2d(capacity)
        self.bn2 = nn.BatchNorm2d(capacity)
        self.bn3 = nn.BatchNorm2d(capacity)
        self.bn4 = nn.BatchNorm2d(capacity * 2)
        self.bn5 = nn.BatchNorm2d(capacity * 16)

        self.bn3_1 = nn.BatchNorm2d(capacity*8)
        self.bn3_2 = nn.BatchNorm2d(capacity*4)
        self.bn3_3 = nn.BatchNorm2d(capacity*2)
        self.bn3_4 = nn.BatchNorm2d(capacity*2)


    def forward(self, point_cloud):

        """ Classification PointNet, input is BxNx3, output BxNx50 """

        # point_cloud [bz, num_cloud, feature_dim]

        num_point = point_cloud.shape[1]
        point_cloud = point_cloud.permute(0,2,1)
        point_cloud = torch.unsqueeze(point_cloud, dim = 3)

        net = self.input_transform(point_cloud)
        net = self.conv1(net)
        net = self.bn1(net)
        net = nn.functional.leaky_relu(net, 0.2)

        net = self.conv2(net)
        net = self.bn2(net)
        point_feature = nn.functional.leaky_relu(net, 0.2)

        net = self.conv3(point_feature)
        net = self.bn3(net)
        net = nn.functional.leaky_relu(net, 0.2)

        net = self.conv4(net)
        net = self.bn4(net)
        net = nn.functional.leaky_relu(net, 0.2)

        net = self.conv5(net)
        net = self.bn5(net)
        net = nn.functional.leaky_relu(net, 0.2)

        global_feat = nn.functional.max_pool2d(net, [num_point, 1])
        print(global_feat)

        global_feat_expand = global_feat.expand(-1, -1, num_point, -1)
        concat_feat = torch.cat((point_feature, global_feat_expand), 1)
        print(concat_feat)

        net = self.conv3_1(concat_feat)
        net = self.bn3_1(net)
        net = nn.functional.leaky_relu(net, 0.2)
        net = self.conv3_2(net)
        net = self.bn3_2(net)
        net = nn.functional.leaky_relu(net, 0.2)
        net = self.conv3_3(net)
        net = self.bn3_3(net)
        net = nn.functional.leaky_relu(net, 0.2)
        net = self.conv3_4(net)
        net = self.bn3_4(net)
        net = nn.functional.leaky_relu(net, 0.2)
        net = self.conv3_5(net)
        net = net.squeeze(2)  # BxNxC
        return net




class input_transform_net(nn.module):
    def __init__(self, capacity = CAPACITY, k = 3):
        assert (k == 3)
        super(input_transform_net, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_FEATURE_DIM, capacity, 1)
        self.conv2 = nn.Conv2d(capacity, capacity*2, 1)
        self.conv3 = nn.Conv2d(capacity, capacity*16, 1)

        self.conv4 = nn.Conv2d(capacity * 16, capacity * 8, 1)
        self.conv5 = nn.Conv2d(capacity * 8, capacity * 4, 1)

        self.bn1 = nn.BatchNorm2d(capacity)
        self.bn2 = nn.BatchNorm2d(capacity*2)
        self.bn3 = nn.BatchNorm2d(capacity*16)
        self.bn4 = nn.BatchNorm2d(capacity*8)
        self.bn5 = nn.BatchNorm2d(capacity*4)

        self.conv_k = nn.Conv2d(capacity * 4, 3 * k, 1)
        init_weight = torch.Tensor(np.zeros([capacity * 4, 3 * k], dtype= np.float32))
        init_weight = nn.Parameter(init_weight)

        init_biase = np.array([1,0,0,0,1,0,0,0,1], dtype = np.float32)
        init_biase = torch.Tensor(init_biase)
        init_biase = nn.Parameter(init_biase)

        self.conv_k.weight = init_weight
        self.conv_k.bias = init_biase


    def forward(self, pointcloud_xyz):
        num_point = pointcloud_xyz.shape(2)
        batch_size = pointcloud_xyz.shape(0)
        k_dim = pointcloud_xyz.shape(1)
        net = self.conv1(pointcloud_xyz)
        net = self.bn1(net)
        net = self.conv2(net)
        net = self.bn2(net)
        net = self.conv3(net)
        net = self.bn3(net)
        net = self.conv4(net)
        net = self.bn4(net)
        net = self.conv5(net)
        net = self.bn5(net)
        transform_mat = self.conv_k(net)
        transform_mat = torch.reshape(transform_mat, (batch_size, num_point, k_dim, 3))#shape(B, num, k, 3)

        pointcloud_xyz = pointcloud_xyz.purmute(0,2,3,1) #shape(B, num, 1, k)
        out_xyz = torch.matmul(pointcloud_xyz, transform_mat)#shape(B, num, 1, 3)
        out_xyz = out_xyz.purmute(0, 3, 1, 2)
        return out_xyz






