from network import Network
import tensorflow as tf

class ICNet(Network):
    def setup(self, is_training, num_classes):
        (self.feed('data')
             .interp(factor=0.5, name='data_sub2')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv1_1_3x3_s2')  #1/4
             .conv(3, 3, 32, 1, 1, biased=True, padding='SAME', relu=True, name='conv1_2_3x3')
             .conv(3, 3, 64, 1, 1, biased=True, padding='SAME', relu=True, name='conv1_3_3x3')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_1_1x1_proj'))     #1/8

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_1_1x1_reduce')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_1_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_1_1x1_increase'))

        (self.feed('conv2_1_1x1_proj',
                   'conv2_1_1x1_increase')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_2_1x1_reduce')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_2_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_2_1x1_increase'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_3_1x1_reduce')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_3_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_3_1x1_increase'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 256, 2, 2, biased=True, relu=False, name='conv3_1_1x1_proj'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 64, 2, 2, biased=True, relu=True, name='conv3_1_1x1_reduce')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_1_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_1_1x1_increase'))

        (self.feed('conv3_1_1x1_proj',
                   'conv3_1_1x1_increase')                                              #1/16
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .interp(factor=0.5, name='conv3_1_sub4')                                   #1/32
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_2_1x1_reduce')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_2_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_2_1x1_increase'))

        (self.feed('conv3_1_sub4',
                   'conv3_2_1x1_increase')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_3_1x1_reduce')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_3_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_3_1x1_increase'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_4_1x1_reduce')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_4_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_4_1x1_increase'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_1_1x1_proj'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_1_1x1_reduce')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_1_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_1_1x1_increase'))

        (self.feed('conv4_1_1x1_proj',
                   'conv4_1_1x1_increase')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_2_1x1_reduce')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_2_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_2_1x1_increase'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_3_1x1_reduce')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_3_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_3_1x1_increase'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_4_1x1_reduce')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_4_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_4_1x1_increase'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_5_1x1_reduce')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_5_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_5_1x1_increase'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_6_1x1_reduce')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_6_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_6_1x1_increase'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_1_1x1_proj'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_1_1x1_reduce')
             .zero_padding(paddings=4, name='padding14')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_1_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_1_1x1_increase'))

        (self.feed('conv5_1_1x1_proj',
                   'conv5_1_1x1_increase')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_2_1x1_reduce')
             .zero_padding(paddings=4, name='padding15')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_2_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_2_1x1_increase'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_3_1x1_reduce')
             .zero_padding(paddings=4, name='padding16')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_3_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_3_1x1_increase'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        shape = self.layers['conv5_3/relu'].get_shape().as_list()[1:3]
        h,w=shape
        (self.feed('conv5_3/relu')
             .avg_pool(h, w, h, w, name='conv5_3_pool1')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/2, w/2, h/2, w/2, name='conv5_3_pool2')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/3, w/3, h/3, w/3, name='conv5_3_pool3')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/4, w/4, h/4, w/4, name='conv5_3_pool6')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .add(name='conv5_3_sum')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_4_k1')
             .interp(factor=2.0, name='conv5_4_interp')
             .zero_padding(paddings=2, name='padding17')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=False, name='conv_sub4'))

        (self.feed('conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv3_1_sub2_proj'))

        (self.feed('conv_sub4',
                   'conv3_1_sub2_proj')
             .add(name='sub24_sum')
             .relu(name='sub24_sum/relu')
             .interp(factor=2.0, name='sub24_sum_interp')
             .zero_padding(paddings=2, name='padding18')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=False, name='conv_sub2'))

        (self.feed('data')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv1_sub1')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv2_sub1')
             .conv(3, 3, 64, 2, 2, biased=True, padding='SAME', relu=True, name='conv3_sub1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv3_sub1_proj'))

        (self.feed('conv_sub2',
                   'conv3_sub1_proj')
             .add(name='sub12_sum')
             .relu(name='sub12_sum/relu')
             .interp(factor=2.0, name='sub12_sum_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6_cls')  #1/4
             .interp(factor=2.0, name='conv6_resize')
             .interp(factor=2.0, name='conv6_resize2'))
        # (self.feed('conv5_4_interp')
        #      .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='sub4_out'))
        #
        # (self.feed('sub24_sum_interp')
        #      .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='sub24_out'))

        (self.feed('conv5_4_k1')
             .max_pool(4,2,4,2, name='img_semantic_64'))




class PC_MaxPool_Net(Network):
    def setup(self, is_training, num_classes):
        (self.feed('mask_map')
             .resize_nearest([512,512],name='nearst1'))
             #.tile([1,1,1,35],name='35_mask'))





        (self.feed('pc_map')
             .resize_nearest([512,512],name='pc_map_pool'))

        #
        (self.feed('img_features',
                   'pc_map_pool')
             .concat(name='img_features_plusxyz',axis=-1))

        (self.feed('img_features_plusxyz',
                   'nearst1')
             .element_multi(name='mixd_feature_mask')
             .conv(1,1,64,1,1,biased=False, relu=True,name='conv_p_2')
             #.batch_normalization(relu=True, name='conv_p_2_bn')
             .conv(1,1,64,1,1, biased=False, relu=True, name='conv_p_3')
             # .batch_normalization(relu=True, name='conv_p_3_bn')
             .conv(1,1,64,1,1,biased=False, relu=True,name='conv_p_4')
             # .batch_normalization(relu=True, name='conv_p_4_bn')
             .conv(1,1,128,1,1,biased=False, relu=True,name='conv_p_5')
             #.batch_normalization(relu=True, name='conv1_p_5_bn')
             .max_pool(64,32,64,32,name='pc_maxpool'))

        (self.feed('pc_maxpool',
                   'img_semantic_64')
             .concat(name='maxpool_concat_img_highfeat',axis=-1)
             .conv(1,1,128,1,1, biased=True, padding='SAME', relu=True, name='pc_feat1')
             .conv(1,1,128,1,1,biased=True, relu=True,name='pc_feat2')
             #.batch_normalization(relu=True, name='pc_feat1_bn')
             .conv(1,1,64,1,1,biased=True, relu=True,name='pc_feat3')
             .conv(1, 1,64, 1, 1, biased=True, relu=True, name='pc_feat4')
             #.batch_normalization(relu=True, name='pc_feat2_bn')
             .resize_nearest(shape=[512,512],name='pc_feat4_resize'))

        (self.feed('pc_feat4_resize',
                   'conv_p_2')
             .concat(name='point_pc_feat_concat',axis=-1)
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='predict1'))
             #.batch_normalization(relu=True, name='predict1_bn')
             #.conv(1, 1, 64, 1, 1, biased=True, relu=True, name='predict2'))
             #.batch_normalization(relu=True, name='predict2_bn')

        (self.feed('predict1')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='pc_predict'))

        (self.feed('pc_predict',
                   'nearst1')
             .element_multi(name='pc_predict_mask'))

        (self.feed('img_predict',
                   'pc_predict_mask')
             .add(name='1/2final_predict')
             .interp(factor=2.0, name='final_predict'))


        #
        # (self.feed('pc_predict',
        #            'img_predict')
        #      .concat(name='predict_concat',axis=-1)
        #      .conv(1,1,num_classes,1,1,biased=True,relu=False,name='Gated_Matrix')
        #      .sigmoid(name='Sigmoid_Gated_Matrix'))
        #
        # (self.feed('Sigmoid_Gated_Matrix',
        #            'numclass_mask')
        #      .element_multi(name='Masked_Gated_Matrix'))

class Depth_Feedto_Image_Net(Network):
    def setup(self, is_training, num_classes):
        # (self.feed('mask_map')
        #      .resize_nearest(shape=[512,1024],name='mask_map_pool')
        #      .tile([1,1,1,3],name='3_mask'))

        # (self.feed('mask_map_pool')
        #      .tile([1, 1, 1, num_classes], name='numclass_mask'))

        # (self.feed('img_features')
        #      .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv_p_1')
        #      #.batch_normalization(relu=True, name='conv_p_1_bn')
        #      .interp(factor=2.0, name='img_features_resize'))

        (self.feed('pc_map')
             .resize_nearest(shape=[512,1024], name='pc_map_pool'))

        #
        # (self.feed('img_features_resize',
        #            'pc_map_pool')
        #      .concat(name='img_features_plusxyz',axis=-1))

        (self.feed('pc_map_pool')
             .conv(1,1,64,1,1,biased=False, relu=True,name='conv_p_2')
             #.batch_normalization(relu=True, name='conv_p_2_bn')
             .conv(1,1,64,1,1, biased=False, relu=True, name='conv_p_3')
             # .batch_normalization(relu=True, name='conv_p_3_bn')
             .conv(1,1,64,1,1,biased=False, relu=True,name='conv_p_4')
             # .batch_normalization(relu=True, name='conv_p_4_bn')
             .conv(1,1,512,1,1,biased=False, relu=True,name='conv_p_5')
             #.batch_normalization(relu=True, name='conv1_p_5_bn')
             .max_pool(65,65,8,8,padding='SAME',name='pc_maxpool'))

        (self.feed('pc_maxpool')
             .conv(1, 1,256, 1, 1, biased=True, relu=True, name='pc_feat1')
             .conv(1,1,128,1,1,biased=True, relu=True,name='pc_feat2')
             #.batch_normalization(relu=True, name='pc_feat1_bn')
             .conv(1,1,64,1,1,biased=True, relu=True,name='pc_feat3')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='pc_feat4')
             .zero_padding(paddings=1, name='pnet_padding1')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='pc_conv1')
             .zero_padding(paddings=1, name='pnet_padding2')
             .conv(3, 3, 128, 1, 1, biased=True, relu=True, name='pc_conv2'))
             #.batch_normalization(relu=True, name='pc_feat2_bn'))
        #
        # (self.feed('pc_feat4_resize',
        #            'conv_p_2')
        #      .concat(name='point_pc_feat_concat',axis=-1)
        #      .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='predict1')
        #      #.batch_normalization(relu=True, name='predict1_bn')
        #      .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='predict2'))
             #.batch_normalization(relu=True, name='predict2_bn')

        # (self.feed('predict2')
        #      .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='pc_predict'))
        #
        # (self.feed('pc_predict',
        #            'numclass_mask')
        #      .element_multi(name='pc_predict_mask'))
        #
        # (self.feed('img_predict',
        #            'pc_predict_mask')
        #      .add(name='1/2final_predict')
        #      .interp(factor=2.0, name='final_predict'))


class P_ICNET(Network):
    def setup(self, is_training, num_classes):
        (self.feed('data')
             .interp(factor=0.5, name='data_sub2')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv1_1_3x3_s2')
             .conv(3, 3, 32, 1, 1, biased=True, padding='SAME', relu=True, name='conv1_2_3x3')
             .conv(3, 3, 64, 1, 1, biased=True, padding='SAME', relu=True, name='conv1_3_3x3')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_1_1x1_proj'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_1_1x1_reduce')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_1_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_1_1x1_increase'))

        (self.feed('conv2_1_1x1_proj',
                   'conv2_1_1x1_increase')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_2_1x1_reduce')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_2_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_2_1x1_increase'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_3_1x1_reduce')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_3_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_3_1x1_increase'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 256, 2, 2, biased=True, relu=False, name='conv3_1_1x1_proj'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 64, 2, 2, biased=True, relu=True, name='conv3_1_1x1_reduce')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_1_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_1_1x1_increase'))

        (self.feed('conv3_1_1x1_proj',
                   'conv3_1_1x1_increase')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .interp(factor=0.5, name='conv3_1_sub4')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_2_1x1_reduce')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_2_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_2_1x1_increase'))

        (self.feed('conv3_1_sub4',
                   'conv3_2_1x1_increase')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_3_1x1_reduce')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_3_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_3_1x1_increase'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_4_1x1_reduce')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_4_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_4_1x1_increase'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_1_1x1_proj'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_1_1x1_reduce')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_1_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_1_1x1_increase'))

        (self.feed('conv4_1_1x1_proj',
                   'conv4_1_1x1_increase')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_2_1x1_reduce')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_2_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_2_1x1_increase'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_3_1x1_reduce')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_3_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_3_1x1_increase'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_4_1x1_reduce')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_4_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_4_1x1_increase'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_5_1x1_reduce')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_5_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_5_1x1_increase'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_6_1x1_reduce')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_6_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_6_1x1_increase'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_1_1x1_proj'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_1_1x1_reduce')
             .zero_padding(paddings=4, name='padding14')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_1_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_1_1x1_increase'))

        (self.feed('conv5_1_1x1_proj',
                   'conv5_1_1x1_increase')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_2_1x1_reduce')
             .zero_padding(paddings=4, name='padding15')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_2_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_2_1x1_increase'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_3_1x1_reduce')
             .zero_padding(paddings=4, name='padding16')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_3_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_3_1x1_increase'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        shape = self.layers['conv5_3/relu'].get_shape().as_list()[1:3]

        (self.feed('conv5_3/relu')
             .avg_pool(32, 64, 32, 64, name='conv5_3_pool1')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(16, 32, 16, 32, name='conv5_3_pool2')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(12, 24, 10, 20, name='conv5_3_pool3')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(8, 16, 5, 10, name='conv5_3_pool6')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .add(name='conv5_3_sum')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_4_k1')
             .interp(factor=2.0, name='conv5_4_interp'))
             #.zero_padding(paddings=2, name='padding17'))
             #.atrous_conv(3, 3, 128, 2, biased=True, relu=False, name='conv_sub4'))
        (self.feed('conv5_4_interp',
                   'pnet_input')
             .concat(name='1/16_concat_pnet',axis=-1)
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv_concat_pnet'))




        (self.feed('conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv3_1_sub2_proj'))

        (self.feed('conv_concat_pnet',
                   'conv3_1_sub2_proj')
             .add(name='sub24_sum')
             .relu(name='sub24_sum/relu')
             .interp(factor=2.0, name='sub24_sum_interp')
             .zero_padding(paddings=2, name='padding18')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=False, name='conv_sub2'))

        (self.feed('data')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv1_sub1')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv2_sub1')
             .conv(3, 3, 64, 2, 2, biased=True, padding='SAME', relu=True, name='conv3_sub1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv3_sub1_proj'))

        (self.feed('conv_sub2',
                   'conv3_sub1_proj')
             .add(name='sub12_sum')
             .relu(name='sub12_sum/relu')
             .interp(factor=2.0, name='sub12_sum_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6_cls')  #1/4
             .interp(factor=2.0, name='conv6_resize')
             .interp(factor=2.0, name='conv6_resize2'))