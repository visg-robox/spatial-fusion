"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet
"""

from __future__ import print_function
import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
import shutil
from scipy import misc

from frustum_model import PC_MaxPool_Net,ICNet
from tools import decode_labels,decode_pclabel

from PIL import Image
import cv2
import pointnet_plug
import h5py
sys.path.append('/media/luo/project/project/semantic_segmentation/proposal/ICNET-pointnet/pointnet_seg/sem_seg')
from eval_API import log_string,getaccuracy,eval_print_save

IMG_MEAN = np.array(( 61.90575142,69.36021519,84.3502593 ), dtype=np.float32)

TRIAN_PATH ='/media/luo/project/project/semantic_segmentation/data/dataset/cityscapes/mydata/tfrecord/img_depth/train.tfrecords'
TEST_PATH ='/media/luo/project/project/semantic_segmentation/data/dataset/cityscapes/mydata/tfrecord/img_depth/val.tfrecords'
SAVE_DIR = './model/CRALA_frustum_v2_4_dense'
LOG_DIR = './log/CRALA_frustum_v2_4_dense'
BATCH_SIZE = 1
IGNORE_LABEL = 25
INPUT_SIZE = '1024,1024'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 13  # 27
VALIDCLASS = 13 # 12
NUM_EPOCH = 1
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = ''
SNAPSHOT_DIR = ''

SAVE_PRED_EVERY = 10
LAMBDA1, LAMBDA2, LAMBDA3 = [0, 0, 1]
TRAIN_NUM_ONEEPOCH = 450
TEST_NUM_ONEEPOCH = 50
Initial_step=0
Max_iterstep=5000
End_learning_rate=1e-6

WIDTH=1024
HEIGHT=1024
DECAY_STEP = 450
DECAY_RATE = 0.5

CAPACITY = 200
MIN_DEQUEUE = 30


# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss


def get_arguments():
    parser = argparse.ArgumentParser(description="ICNet")

    parser.add_argument("--data-dir", type=str, default=TRIAN_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--test-dir", type=str, default=TEST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to save graph")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCH,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", action="store_true", default=False,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true", default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true", default=False,
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true", default=False,
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument('--decay_step', type=int, default=DECAY_STEP, help='Decay step for lr decay [default: 5000]')
    parser.add_argument('--decay_rate', type=float, default=DECAY_RATE, help='Decay rate for lr decay [default: 0.5]')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def get_learning_rate(batch):
    learning_rate = tf.train.polynomial_decay(
        LEARNING_RATE,
        batch - Initial_step, Max_iterstep
        , End_learning_rate, power=0.9)
    return learning_rate


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def get_mask(gt, valid_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, valid_classes - 1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    greater_equal = tf.greater_equal(gt, 0)
    mask = tf.logical_and(tf.logical_and(less_equal_class, not_equal_ignore), greater_equal)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices


def create_loss_acc(output, label, num_classes, ignore_label, valid_classes):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = tf.reshape(label, [-1, ])

    indices = get_mask(label, valid_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss


def main():
    """Create the model and start the training."""
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    batch = tf.Variable(0)
    learning_rate = get_learning_rate(batch)
    coord = tf.train.Coordinator()

    with tf.name_scope('placeholder'):
        epoch = tf.placeholder(dtype=tf.float32, shape=())
        is_test = tf.placeholder(tf.bool, shape=())
        image_batch=tf.placeholder(tf.float32,shape=[args.batch_size,h,w,3])
        label_batch=tf.placeholder(tf.int32,shape=[args.batch_size,h,w,1])
        mask_map=tf.placeholder(tf.float32,shape=[args.batch_size,h,w,1])
        pc_map=tf.placeholder(tf.float32,shape=[args.batch_size,h,w,3])

    with tf.variable_scope('ICNET'):
        net = ICNet({'data': image_batch}, is_training=~is_test, num_classes=args.num_classes)
    with tf.variable_scope('PC_MaxPool_Net'):
        net2= PC_MaxPool_Net({'img_features':net.layers['conv1_sub1'],'pc_map':pc_map,'mask_map':mask_map,'img_semantic_64':net.layers['img_semantic_64'],'img_predict':net.layers['conv6_resize']},is_training=~is_test, num_classes=args.num_classes)


    with tf.name_scope('loss'):
        # Gated_Matrix=net2.layers['Masked_Gated_Matrix']
        # Pc_pred_softmax=net2.layers['pc_predict']
        # img_pred_softmax=net.layers['conv6_resize']
        # M_shape=Gated_Matrix.get_shape().as_list()
        # Img_Gated_Matrix=tf.subtract(tf.ones(M_shape),Gated_Matrix)
        # final_pred_softmax=tf.add(tf.multiply(Img_Gated_Matrix,img_pred_softmax),tf.multiply(Gated_Matrix,Pc_pred_softmax),name='Final_score')

        img_pred_softmax = net.layers['conv6_resize']
        final_pred_softmax=net2.layers['final_predict']
        img_pred_softmax = tf.image.resize_bilinear(img_pred_softmax, size=input_size, align_corners=True)

        final_pred_out = tf.argmax(final_pred_softmax, dimension=3)
        final_pred_out = tf.expand_dims(final_pred_out, dim=3)
        img_pred = tf.argmax(img_pred_softmax, dimension=3)
        img_pred = tf.expand_dims(img_pred, dim=3)

        loss_predict = create_loss_acc(final_pred_softmax, label_batch, args.num_classes, args.ignore_label, VALIDCLASS)
        img_loss= create_loss_acc(img_pred_softmax, label_batch, args.num_classes, args.ignore_label, VALIDCLASS)


    with tf.name_scope('variables'):
        all_trainable = [v for v in tf.trainable_variables() if
                         ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]

        #all_trainable=[v for v in all_trainable if 'ICNET' not in v.name]

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        #train_var=[v for v in all_trainable if 'PC_MaxPool_Net' in v.name]
        train_var=all_trainable
        restore_var=[v for v in var_list if 'ICNET' in v.name]
        #restore_var=var_list


        l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]
        reduced_loss = loss_predict + tf.add_n(l2_losses)



    with tf.name_scope('log_loss'):
        log_img_loss=tf.summary.scalar(tensor=img_loss, name='log_img_loss', family='loss')
        log_pnet_loss = tf.summary.scalar(tensor=loss_predict, name='log_pnet_loss', family='loss')
        log_final_loss = tf.summary.scalar(tensor=reduced_loss, name='log_totalloss', family='loss')

        log_img_score=tf.summary.histogram(name='log_img_score',values=img_pred_softmax,family='score')
        log_final_score = tf.summary.histogram( name='log_img_score',values=final_pred_softmax, family='score')
    # with tf.name_scope('accuracy'):
    #     log_acc4=tf.summary.scalar(tensor=acc4,name='accuracy4',family='accuracy')
    #     log_acc24=tf.summary.scalar(tensor=acc24,name='accuracy24',family='accuracy')
    #     log_acc124=tf.summary.scalar(tensor=acc124,name='accuracy124',family='accuracy')

    # with tf.name_scope('histogram'):
    #     log_histogram=tf.summary.histogram(values=sub124_out,name='pred_hisgram',family='histogram')

        log_loss = tf.summary.merge([log_final_loss,log_img_loss,log_final_score,log_img_score])

    with tf.name_scope('recode_img'):
        inputimg = tf.slice(image_batch, [0, 0, 0, 0], [1, h, w, 3], name='sampleimg')
        labelimg = tf.slice(label_batch, [0, 0, 0, 0], [1, h, w, 1], name='labelimg')
        img_pred1=tf.slice(img_pred,[0, 0, 0, 0], [1, h, w, 1],name='imgpred_sample')
        final_pred1 = tf.slice(final_pred_out, [0, 0, 0, 0], [1, h, w, 1], name='finalpred_sample')
        mask=tf.slice(mask_map, [0, 0, 0, 0], [1, h, w, 1], name='mask')
        # Predictions.

        showmask = tf.summary.image('mask', mask)
        showimg = tf.summary.image('img', inputimg)
        showlabel = decode_labels(labelimg, [h, w], args.num_classes)
        showlabel = tf.summary.image('label', showlabel)
        show_imgpred_out = decode_labels(img_pred1, [h, w], args.num_classes)
        show_imgpred = tf.summary.image('img_pred', show_imgpred_out)
        show_finalpred_out = decode_labels(final_pred1, [h, w], args.num_classes)
        show_finalpred = tf.summary.image('final_pred', show_finalpred_out)

        visionlize = tf.summary.merge([showimg, showlabel, show_imgpred,show_finalpred,showmask], name='visionlize')

    # Using Poly learning rate policy
    with tf.name_scope('train_parameter'):
        # base_lr = tf.constant(args.learning_rate)
        # learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - epoch / args.num_epochs), args.power))
        log_lr=tf.summary.scalar(tensor=learning_rate,name='leaning_rate',family='lr')
        # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
        if args.update_mean_var == False:
            update_ops = None
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            #opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
            opt_conv = tf.train.AdamOptimizer(learning_rate, args.momentum)
            grads = tf.gradients(reduced_loss, train_var)
            train_op = opt_conv.apply_gradients(zip(grads, train_var),global_step=batch)
    # Set up tf session and initialize variables.

    log_loss = tf.summary.merge([log_final_loss, log_img_loss,log_lr])

    with tf.name_scope('config'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run([init_g, init_l])
        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

        ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('no_pretrained_model')
            '''
            print('Restore from pre-trained model...')
            net.load(args.restore_from, sess)
            '''
        train_writer = tf.summary.FileWriter(args.log_dir + '/train/', sess.graph)
        eval_train_writer=tf.summary.FileWriter(args.log_dir + '/Eval_train/', sess.graph)
        test_writer = tf.summary.FileWriter(args.log_dir + '/test/', sess.graph)
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    print('start training,max_epoch {:d} \t steps every epoch {:d} \t batch_size {:d} \t learning_rate {:f}'.format(
        args.num_epochs, args.save_pred_every, args.batch_size, args.learning_rate))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    def train_one_epoch(save_loss_step, batch_size, Epoch, istest=False):

        LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
        LOG_FOUT.write(str(args) + '\n')
        total_accuracy = np.zeros(shape=[NUM_CLASSES, 3], dtype=np.float32)
        loss_sum = 0
        writer = train_writer
        front = 'train'
        log_string(LOG_FOUT, ('**** EPOCH %03d ****' + front + '*****') % (Epoch))
        sys.stdout.flush()

        Infer_time=0
        IMG_PATH='/media/luo/project/project/semantic_segmentation/data/dataset/CARLA/CARLA_/PythonClient/out_lidar_camera_semantic/episode_17/Data/list.txt'
        log_time=open('time.txt', 'w')

        with open(IMG_PATH) as f_imglist:
            imglist = [line.rstrip() for line in f_imglist]
            num_one_epoch = len(imglist)
            num_batches = num_one_epoch // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_img_path=imglist[start_idx:end_idx]
            img_data=np.zeros([batch_size,HEIGHT,WIDTH,3])
            pc_map_data = np.zeros([batch_size, HEIGHT, WIDTH, 3])
            mask_data = np.zeros([batch_size, HEIGHT, WIDTH])
            label_data= np.zeros([batch_size, HEIGHT, WIDTH])
            for i in range(batch_size):
                img_data[i]=np.array(Image.open(batch_img_path[i]),dtype=np.float32)
                label_data[i]=np.array(cv2.imread(batch_img_path[i].replace('RGB','Semantic'),-1))[:,:,2]
                pc_map_data[i]=np.load(batch_img_path[i].replace('RGB','Lidar_depth').replace('png','npy'))

                mask_data[i]=np.zeros_like(label_data[i])
                mask_data[i][np.where(pc_map_data[i][:,:,1])]=1

            img_data-=IMG_MEAN
            label_data=np.expand_dims(label_data,axis=-1)
            mask_data=np.expand_dims(mask_data,axis=-1)


            feed_dict = {epoch: Epoch, is_test: istest,image_batch:img_data,label_batch:label_data,mask_map:mask_data,pc_map:pc_map_data
}


            if batch_idx % save_loss_step == 0:
                start_time = time.time()
                log_string(LOG_FOUT,'Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
                showloss, loss_value,show_vision,_ = sess.run(
                    [log_loss, reduced_loss,visionlize,train_op],
                    feed_dict=feed_dict)

                writer.add_summary(show_vision, Epoch * num_batches + batch_idx)
                writer.add_summary(showloss, Epoch * num_batches + batch_idx)
                duration = time.time() - start_time
                log_string(LOG_FOUT,'batch loss = {:.3f} ({:.3f} sec/step)'.format(
                    loss_value, duration))
                if loss_value:
                    loss_sum += loss_value

                if 0:
                    total_accuracy += getaccuracy(out, label_data, VALIDCLASS)


            if batch_idx==num_batches-1:
                print('save model and accuracy\n')
                showloss,  loss_value,  _ = sess.run(
                    [log_loss, reduced_loss,
                     train_op],
                    feed_dict=feed_dict)

                writer.add_summary(showloss, Epoch * num_batches + batch_idx)
                #writer.add_summary(visionlizes, Epoch * num_batches/1000 + batch_idx)
                save(saver, sess, args.save_dir, Epoch)
                if 0:
                    total_accuracy += getaccuracy(out, label_data, VALIDCLASS)

                if loss_value:
                    loss_sum += loss_value
                log_string(LOG_FOUT, 'mean loss: %f' % (loss_sum / float(num_batches)))




            if 0:

                time1=time.time()
                pc,img= sess.run([pc_map, image_batch],
                                                  feed_dict=feed_dict)

                time2=time.time()

                output=sess.run([],
                                                  feed_dict=feed_dict)
                time3=time.time()
                infer_time=(time3-time2)-(time2-time1)
                Infer_time+=infer_time
                log_time.write(str(infer_time)+' ')
                misc.imsave(.replace('RGB', 'Infer_sem2'), np.uint8(pred_ID)
                if batch_idx==120:
                    print(Infer_time/120)
                    log_time.close()
        if 0:
            eval_print_save(total_accuracy, 'frustum_link', args.log_dir)



    def eval_one_epoch_train(batch_size, Epoch, istest=True):
        total_accuracy = np.zeros(shape=[NUM_CLASSES, 3], dtype=np.float32)
        loss_sum = 0
        writer=eval_train_writer
        front = 'train'
        Dir=os.path.join(args.log_dir,'Eval_train')
        if not os.path.exists(Dir):
            os.makedirs(Dir)

        LOG_FOUT=open(os.path.join(Dir,'log.txt'),'a')
        log_string(LOG_FOUT, ('**** EPOCH %03d ****' + front + '*****') % (Epoch))
        sys.stdout.flush()

        IMG_PATH = '/media/luo/project/project/semantic_segmentation/data/dataset/SYNTHIA/pc_mask_map/img_trainlist.txt'
        PC_MAP_PATH = '/media/luo/project/project/semantic_segmentation/data/dataset/SYNTHIA/pc_mask_map/pc_map_trainlist.txt'

        with open(IMG_PATH) as f_imglist:
            with open(PC_MAP_PATH) as f_pc_map:
                imglist = [line.rstrip() for line in f_imglist]
                pc_map_list = [line.rstrip() for line in f_pc_map]
                num_one_epoch=len(imglist)
                num_batches = num_one_epoch // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_img_path = imglist[start_idx:end_idx]
            batch_pc_map_path = pc_map_list[start_idx:end_idx]
            img_data = np.zeros([batch_size,752, WIDTH, 3])
            pc_map_data = np.zeros([batch_size, 752, WIDTH, 3])
            mask_data = np.zeros([batch_size, 752, WIDTH])
            label_data = np.zeros([batch_size, 752, WIDTH])
            for i in range(batch_size):
                img_data[i] = np.array(Image.open(batch_img_path[i]))[0:752, :]
                label_data[i] = np.array(cv2.imread(batch_img_path[i].replace('RGB', 'GT/LABELS'), -1))[0:752, :, 2]
                pc_temp = np.load(batch_pc_map_path[i])
                pc_map_data[i] = pc_temp[:, :, 1:]
                mask_data[i] = pc_temp[:, :, 0]
            img_data -= IMG_MEAN
            label_data = np.expand_dims(label_data, axis=-1)
            mask_data = np.expand_dims(mask_data, axis=-1)

            img_data = np.pad(img_data, ((0, 0), (8, 8), (0,0), (0,0)), 'constant')
            label_data = np.pad(label_data, ((0, 0), (8, 8), (0,0), (0,0)), 'constant')
            pc_map_data = np.pad(pc_map_data, ((0, 0), (8, 8), (0,0), (0,0)), 'constant')
            mask_data = np.pad(mask_data, ((0, 0), (8, 8), (0,0), (0,0)), 'constant')

            label_data -= 1

            feed_dict = {epoch: Epoch, is_test: istest, image_batch: img_data, label_batch: label_data,
                         mask_map: mask_data, pc_map: pc_map_data
                         }

            if batch_idx % 500 == 0:
                start_time = time.time()
                log_string(LOG_FOUT, 'Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
                # loss_value, out,decode_fetch,decode_img_fetch= sess.run(
                #     [reduced_loss, final_pred_out,show_finalpred_out,show_imgpred_out],
                #     feed_dict=feed_dict)



                loss_value,out,visionlize_fetch,log = sess.run(
                         [reduced_loss, final_pred_out,visionlize,log_loss],
                         feed_dict=feed_dict)
                writer.add_summary(log, Epoch * num_batches / 1000 + batch_idx)
                writer.add_summary(visionlize_fetch, Epoch * num_batches / 1000 + batch_idx)
                duration = time.time() - start_time
                print('total loss = {:.3f} ({:.3f} sec/step)'.format(
                    loss_value, duration))
                loss_sum += loss_value
                total_accuracy += getaccuracy(out, label_data, VALIDCLASS)
                # misc.imsave(os.path.join(Dir, 'Test_Output', str(batch_idx) + '.png'), decode_fetch[0])
                # misc.imsave(os.path.join(Dir, 'Test_img_Output', str(batch_idx) + '.png'),decode_img_fetch[0])
                # shutil.copyfile(imglist[batch_idx].replace('RGB', 'Depth'), os.path.join(Dir,'Depth','depth'+str(batch_idx) + '.png'))


            else:
                out,log = sess.run([final_pred_out,log_loss],
                                              feed_dict=feed_dict)
                writer.add_summary(log, Epoch * num_batches / 1000 + batch_idx)

                total_accuracy += getaccuracy(out, label_data, VALIDCLASS)

        eval_print_save(total_accuracy, 'img_result', Dir)
        LOG_FOUT.close()




    def eval_one_epoch_test(batch_size, Epoch, istest=True):
        total_accuracy = np.zeros(shape=[NUM_CLASSES, 3], dtype=np.float32)
        loss_sum = 0
        writer=test_writer
        front = 'test'
        Dir=os.path.join(args.log_dir,'Eval_test')
        if not os.path.exists(Dir):
            os.makedirs(Dir)

        LOG_FOUT=open(os.path.join(Dir,'log.txt'),'a')
        log_string(LOG_FOUT, ('**** EPOCH %03d ****' + front + '*****') % (Epoch))
        sys.stdout.flush()

        IMG_PATH = '/media/luo/project/project/semantic_segmentation/data/dataset/SYNTHIA/pc_mask_map/img_testlist.txt'
        PC_MAP_PATH = '/media/luo/project/project/semantic_segmentation/data/dataset/SYNTHIA/pc_mask_map/pc_map_testlist.txt'

        with open(IMG_PATH) as f_imglist:
            with open(PC_MAP_PATH) as f_pc_map:
                imglist = [line.rstrip() for line in f_imglist]
                pc_map_list = [line.rstrip() for line in f_pc_map]
                num_one_epoch=len(imglist)
                num_batches = num_one_epoch // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_img_path = imglist[start_idx:end_idx]
            batch_pc_map_path = pc_map_list[start_idx:end_idx]
            img_data = np.zeros([batch_size,752, WIDTH, 3])
            pc_map_data = np.zeros([batch_size, 752, WIDTH, 3])
            mask_data = np.zeros([batch_size, 752, WIDTH])
            label_data = np.zeros([batch_size, 752, WIDTH])
            for i in range(batch_size):
                img_data[i] = np.array(Image.open(batch_img_path[i]))[0:752, :]
                label_data[i] = np.array(cv2.imread(batch_img_path[i].replace('RGB', 'GT/LABELS'), -1))[0:752, :, 2]
                pc_temp = np.load(batch_pc_map_path[i])
                pc_map_data[i] = pc_temp[:, :, 1:]
                mask_data[i] = pc_temp[:, :, 0]
            img_data -= IMG_MEAN
            label_data = np.expand_dims(label_data, axis=-1)
            mask_data = np.expand_dims(mask_data, axis=-1)

            img_data = np.pad(img_data, ((0, 0), (8, 8), (0,0), (0,0)), 'constant')
            label_data = np.pad(label_data, ((0, 0), (8, 8), (0,0), (0,0)), 'constant')
            pc_map_data = np.pad(pc_map_data, ((0, 0), (8, 8), (0,0), (0,0)), 'constant')
            mask_data = np.pad(mask_data, ((0, 0), (8, 8), (0,0), (0,0)), 'constant')

            label_data -= 1

            feed_dict = {epoch: Epoch, is_test: istest, image_batch: img_data, label_batch: label_data,
                         mask_map: mask_data, pc_map: pc_map_data
                         }

            if batch_idx % 500 == 0:
                start_time = time.time()
                log_string(LOG_FOUT, 'Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
                loss_value, out,decode_fetch,decode_img_fetch,visionlize_fetch,log= sess.run(
                    [reduced_loss, final_pred_out,show_finalpred_out,show_imgpred_out,visionlize,log_loss],
                    feed_dict=feed_dict)

                writer.add_summary(log, Epoch * num_batches / 1000 + batch_idx)
                writer.add_summary(visionlize_fetch, Epoch * num_batches / 1000 + batch_idx)
                duration = time.time() - start_time
                print('total loss = {:.3f} ({:.3f} sec/step)'.format(
                    loss_value, duration))
                loss_sum += loss_value
                total_accuracy += getaccuracy(out, label_data, VALIDCLASS)

                if not os.path.exists(os.path.join(Dir, 'Test_Output')):
                    os.makedirs(os.path.join(Dir, 'Test_Output'))
                if not os.path.exists(os.path.join(Dir, 'Test_img_Output')):
                    os.makedirs(os.path.join(Dir, 'Test_img_Output'))
                if not os.path.exists(os.path.join(Dir, 'RGB')):
                    os.makedirs(os.path.join(Dir, 'RGB'))


                misc.imsave(os.path.join(Dir, 'Test_Output', str(batch_idx) + '.png'), decode_fetch[0])
                misc.imsave(os.path.join(Dir, 'Test_img_Output', str(batch_idx) + '.png'),decode_img_fetch[0])
                shutil.copyfile(imglist[batch_idx], os.path.join(Dir,'RGB',str(batch_idx) + '.png'))


            else:
                out,log = sess.run([final_pred_out,log_loss],
                                              feed_dict=feed_dict)
                writer.add_summary(log, Epoch * num_batches / 1000 + batch_idx)

                total_accuracy += getaccuracy(out, label_data, VALIDCLASS)

        eval_print_save(total_accuracy, 'img_result', Dir)
        LOG_FOUT.close()
    # train

    for epo in range(args.num_epochs):
        train_one_epoch(args.save_pred_every, args.batch_size, epo)
        #eval_one_epoch_train(args.batch_size, epo)
        #eval_one_epoch_test(args.batch_size, epo)


    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
