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
from model import ICNet_BN, ICNet
from tools import decode_labels, prepare_label
import image_reader
import xlwt

IMG_MEAN = np.array(( 61.90575142,69.36021519,84.3502593 ), dtype=np.float32)

TRIAN_PATH = 'Data/episode18_2/train.tfrecords'
TEST_PATH = 'Data/episode18_2/val.tfrecords'
SAVE_DIR = './model/CRALA_episode18_2/'
LOG_DIR = './log/CRALA_episode18_2/'
BATCH_SIZE = 3
IGNORE_LABEL = 25
INPUT_SIZE = '1024,1024'
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 13  # 27
VALIDCLASS = 13 # 12
NUM_EPOCH = 20
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = ''
SNAPSHOT_DIR = './model/CRALA_episode18'

SAVE_PRED_EVERY = 10
LAMBDA1, LAMBDA2, LAMBDA3 = [0, 0, 1]
Initial_step=0
TRAIN_NUM_ONEEPOCH = 1380
TEST_NUM_ONEEPOCH = 10

DECAY_STEP = 1600
DECAY_RATE = 0.5
Max_iterstep=10000
End_learning_rate=1e-6


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
    parser.add_argument("--update-mean-var", action="store_true", default=True,
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true", default=True,
                        help="whether to train beta & gamma in bn layer")

    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


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


def get_learning_rate(batch):
    learning_rate = tf.train.polynomial_decay(
        LEARNING_RATE,
        batch - Initial_step, Max_iterstep
        , End_learning_rate, power=0.9)
    return learning_rate

def log_string(writer, out_str):
    writer.write(out_str + '\n')
    writer.flush()
    print(out_str)


def getaccuracy(pred, gt, validclass):
    class_num = np.zeros([validclass, 3], dtype=np.float32)
    for i in range(validclass):
        class_num[i, 0] = np.sum(np.logical_and(np.equal(pred, i), np.greater_equal(gt, 0)))
        class_num[i, 1] = np.sum(np.logical_and(np.equal(gt, i), np.equal(gt, pred)))
        class_num[i, 2] = np.sum(np.equal(gt, i))
    return class_num


def create_loss_acc(output, label, num_classes, ignore_label, valid_classes):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
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
    epoch = tf.placeholder(dtype=tf.float32, shape=())
    is_test = tf.placeholder(tf.bool, shape=())
    with tf.name_scope("create_inputs"):

        image_batch,label_batch = image_reader.datafromtfrecord(args.data_dir, input_size, IMG_MEAN,
                                                                             args.batch_size, CAPACITY, MIN_DEQUEUE)
    #     test_image_batch, test_label_batch = image_reader.datafromtfrecord(args.test_dir, input_size, IMG_MEAN,
    #                                                                        args.batch_size, CAPACITY, MIN_DEQUEUE)

    # image_batch, label_batch = tf.cond(is_test, lambda: [test_image_batch, test_label_batch],
    #                                    lambda: [train_image_batch, train_label_batch])

    with tf.variable_scope('ICNET'):
        net = ICNet_BN({'data': image_batch}, is_training=~is_test, num_classes=args.num_classes)

    with tf.name_scope('loss'):
        sub4_out = net.layers['sub4_out']
        sub24_out = net.layers['sub24_out']
        sub124_out = net.layers['conv6_cls']
        pred_out = tf.image.resize_bilinear(sub124_out, size=input_size, align_corners=True)
        pred_out = tf.argmax(pred_out, dimension=3)
        pred_out = tf.expand_dims(pred_out, dim=3)
        loss_sub4 = create_loss_acc(sub4_out, label_batch, args.num_classes, args.ignore_label, VALIDCLASS)
        loss_sub24 = create_loss_acc(sub24_out, label_batch, args.num_classes, args.ignore_label, VALIDCLASS)
        loss_sub124 = create_loss_acc(sub124_out, label_batch, args.num_classes, args.ignore_label, VALIDCLASS)

    with tf.name_scope('variables'):
        all_trainable = [v for v in tf.trainable_variables() if
                         ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        restore_var = [v for v in var_list if ('ICNET' in v.name)]
        print(restore_var)

        l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in restore_var if 'weights' in v.name]
        reduced_loss = LAMBDA1 * loss_sub4 + LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    with tf.name_scope('log_loss'):
        log_loss4 = tf.summary.scalar(tensor=loss_sub4, name='log_loss4', family='loss')
        log_loss24 = tf.summary.scalar(tensor=loss_sub24, name='log_loss24', family='loss')
        log_loss124 = tf.summary.scalar(tensor=loss_sub124, name='log_loss124', family='loss')
        log_totalloss = tf.summary.scalar(tensor=reduced_loss, name='log_totalloss', family='loss')
    # with tf.name_scope('accuracy'):
    #     log_acc4=tf.summary.scalar(tensor=acc4,name='accuracy4',family='accuracy')
    #     log_acc24=tf.summary.scalar(tensor=acc24,name='accuracy24',family='accuracy')
    #     log_acc124=tf.summary.scalar(tensor=acc124,name='accuracy124',family='accuracy')

    # with tf.name_scope('histogram'):
    #     log_histogram=tf.summary.histogram(values=sub124_out,name='pred_hisgram',family='histogram')

    log_loss = tf.summary.merge([log_totalloss, log_loss4, log_loss24, log_loss124])

    with tf.name_scope('recode_img'):
        inputimg = tf.slice(image_batch, [0, 0, 0, 0], [1, h, w, 3], name='sampleimg')
        labelimg = tf.slice(label_batch, [0, 0, 0, 0], [1, h, w, 1], name='labelimg')
        pred = tf.slice(pred_out, [0, 0, 0, 0], [1, h, w, 1], name='samplepred')

        # Predictions.

        showimg = tf.summary.image('img', inputimg)
        showlabel = decode_labels(labelimg, [h, w], args.num_classes)
        showlabel = tf.summary.image('label', showlabel)
        showpred = decode_labels(pred, [h, w], args.num_classes)
        showpred = tf.summary.image('pred', showpred)
        visionlize = tf.summary.merge([showimg, showlabel, showpred], name='visionlize')

    # Using Poly learning rate policy
    with tf.name_scope('train_parameter'):
        # base_lr = tf.constant(args.learning_rate)
        # learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - epoch / args.num_epochs), args.power))

        # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
        if args.update_mean_var == False:
            update_ops = None
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
            grads = tf.gradients(reduced_loss, all_trainable)
            train_op = opt_conv.apply_gradients(zip(grads, all_trainable), global_step=batch)
    # Set up tf session and initialize variables.

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
        test_writer = tf.summary.FileWriter(args.log_dir + '/test/', sess.graph)
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    print('start training,max_epoch {:d} \t steps every epoch {:d} \t batch_size {:d} \t learning_rate {:f}'.format(
        args.num_epochs, args.save_pred_every, args.batch_size, args.learning_rate))
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(args) + '\n')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    def train_one_epoch(num_one_epoch, save_loss_step, batch_size, Epoch, istest=False):
        total_accuracy = np.zeros(shape=[NUM_CLASSES, 3], dtype=np.float32)
        loss_sum = 0
        num_batches = num_one_epoch // batch_size
        writer = train_writer
        front = 'train'
        log_string(LOG_FOUT, ('**** EPOCH %03d ****' + front + '*****') % (Epoch))
        sys.stdout.flush()

        for batch_idx in range(num_batches):
            if batch_idx % save_loss_step == 0:
                start_time = time.time()
                log_string(LOG_FOUT, 'Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
                showloss, loss_value, loss1, loss2, loss3, out, Gt, _ = sess.run(
                    [log_loss, reduced_loss, loss_sub4, loss_sub24, loss_sub124, pred_out, label_batch, train_op],
                    feed_dict={epoch: Epoch, is_test: istest})

                writer.add_summary(showloss, Epoch * num_batches + batch_idx)
                duration = time.time() - start_time
                log_string(LOG_FOUT,
                           'total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f} ({:.3f} sec/step)'.format(
                               loss_value, loss1, loss2, loss3, duration))

                total_accuracy += getaccuracy(out, Gt, VALIDCLASS)
                loss_sum += loss_value

            if batch_idx == (num_batches - 1):
                log_string(LOG_FOUT, 'epoch end,save model and accuracy\n')
                showloss, visionlizes, loss_value, loss1, loss2, loss3, out, Gt, _ = sess.run(
                    [log_loss, visionlize, reduced_loss, loss_sub4, loss_sub24, loss_sub124, pred_out, label_batch,
                     train_op],
                    feed_dict={epoch: Epoch, is_test: istest})

                writer.add_summary(showloss, Epoch * num_batches + batch_idx)
                writer.add_summary(visionlizes, Epoch)
                save(saver, sess, args.save_dir, Epoch)

                total_accuracy += getaccuracy(out, Gt, VALIDCLASS)
                loss_sum += loss_value

                per_class_accuracy = total_accuracy[:, 1] / total_accuracy[:, 2]
                mean_accuracy = np.sum(total_accuracy[:, 1]) / np.sum(total_accuracy[:, 2])
                per_class_iou = total_accuracy[:, 1] / (
                        total_accuracy[:, 0] + total_accuracy[:, 2] - total_accuracy[:, 1])
                miou = np.sum(total_accuracy[:, 1]) / (
                        np.sum(total_accuracy[:, 0]) + np.sum(total_accuracy[:, 2]) - np.sum(total_accuracy[:, 1]))

                classname = ['void','Buildings','Fences','Other','Pedestrians','Poles','RoadLines','Roads','Sidewalks','Vegetation','Vehicles','Walls','TrafficSigns']
                workbook = xlwt.Workbook(encoding='utf-8')
                booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
                booksheet.write(1, 0, 'accuracy')
                booksheet.write(1, 1, 'train')

                booksheet.write(4, 0, 'iou')
                booksheet.write(4, 1, 'train')

                booksheet.write(7, 0, 'data_distribution')
                for i in range(len(classname)):
                    booksheet.write(0, i + 2, classname[i])
                    booksheet.write(1, i + 2, format(per_class_accuracy[i], '.5'))
                    booksheet.write(4, i + 2, format(per_class_iou[i], '.5'))
                    booksheet.write(7, i + 2, format(total_accuracy[i, 2] / np.sum(total_accuracy[:, 2]), '.5'))
                booksheet.write(0, 2 + 12, 'total')
                booksheet.write(1, 2 + 12, format(mean_accuracy, '.5'))
                booksheet.write(4, 2 + 12, format(miou, '.5'))

                workbook.save(args.log_dir + '/' + 'train' + '_xlwt.xls')

                log_string(LOG_FOUT, 'mean loss: %f' % (loss_sum / float(num_batches)))
                log_string(LOG_FOUT, 'perclass accuracy:')
                log_string(LOG_FOUT, str(per_class_accuracy))
                log_string(LOG_FOUT, 'total_accuracy: %f' % (mean_accuracy))
                log_string(LOG_FOUT, 'perclass iou:')
                log_string(LOG_FOUT, str(per_class_iou))
                log_string(LOG_FOUT, 'miou: %f' % (miou))
                log_string(LOG_FOUT, '')




            else:
                loss_value, out, Gt, _ = sess.run([reduced_loss, pred_out, label_batch, train_op],
                                                  feed_dict={epoch: Epoch, is_test: istest})

                total_accuracy += getaccuracy(out, Gt, VALIDCLASS)
                loss_sum += loss_value

    def eval_one_epoch(num_one_epoch, save_loss_step, batch_size, Epoch, istest=True):
        total_accuracy = np.zeros(shape=[NUM_CLASSES, 3], dtype=np.float32)
        loss_sum = 0
        num_batches = num_one_epoch // batch_size
        writer = test_writer
        front = 'test'
        log_string(LOG_FOUT, ('**** EPOCH %03d ****' + front + '*****') % (Epoch))
        sys.stdout.flush()

        for batch_idx in range(num_batches):
            if batch_idx % save_loss_step == 0:
                start_time = time.time()
                log_string(LOG_FOUT, 'Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
                showloss, loss_value, loss1, loss2, loss3, out, Gt = sess.run(
                    [log_loss, reduced_loss, loss_sub4, loss_sub24, loss_sub124, pred_out, label_batch],
                    feed_dict={epoch: Epoch, is_test: istest})

                writer.add_summary(showloss, Epoch * num_batches + batch_idx)
                duration = time.time() - start_time
                log_string(LOG_FOUT,
                           'total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f} ({:.3f} sec/step)'.format(
                               loss_value, loss1, loss2, loss3, duration))

                total_accuracy += getaccuracy(out, Gt, VALIDCLASS)
                loss_sum += loss_value

            if batch_idx == (num_batches - 1):
                log_string(LOG_FOUT, 'epoch end,save model and accuracy\n')
                showloss, visionlizes, loss_value, loss1, loss2, loss3, out, Gt = sess.run(
                    [log_loss, visionlize, reduced_loss, loss_sub4, loss_sub24, loss_sub124, pred_out,
                     label_batch],
                    feed_dict={epoch: Epoch, is_test: istest})

                writer.add_summary(showloss, Epoch * num_batches + batch_idx)
                writer.add_summary(visionlizes, Epoch)

                total_accuracy += getaccuracy(out, Gt, VALIDCLASS)
                loss_sum += loss_value

                per_class_accuracy = total_accuracy[:, 1] / total_accuracy[:, 2]
                mean_accuracy = np.sum(total_accuracy[:, 1]) / np.sum(total_accuracy[:, 2])
                per_class_iou = total_accuracy[:, 1] / (
                        total_accuracy[:, 0] + total_accuracy[:, 2] - total_accuracy[:, 1])
                miou = np.sum(total_accuracy[:, 1]) / (
                        np.sum(total_accuracy[:, 0]) + np.sum(total_accuracy[:, 2]) - np.sum(
                    total_accuracy[:, 1]))

                classname = ['void','Buildings', 'Fences', 'Other', 'Pedestrians', 'Poles', 'RoadLines', 'Roads', 'Sidewalks',
                             'Vegetation', 'Vehicles', 'Walls', 'TrafficSigns']
                workbook = xlwt.Workbook(encoding='utf-8')
                booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
                booksheet.write(1, 0, 'accuracy')
                booksheet.write(1, 1, 'test')

                booksheet.write(4, 0, 'iou')
                booksheet.write(4, 1, 'test')

                booksheet.write(7, 0, 'data_distribution')
                for i in range(len(classname)):
                    booksheet.write(0, i + 2, classname[i])
                    booksheet.write(1, i + 2, format(per_class_accuracy[i], '.5'))
                    booksheet.write(4, i + 2, format(per_class_iou[i], '.5'))
                    booksheet.write(7, i + 2, format(total_accuracy[i, 2] / np.sum(total_accuracy[:, 2]), '.5'))
                booksheet.write(0, 2 + 12, 'total')
                booksheet.write(1, 2 + 12, format(mean_accuracy, '.5'))
                booksheet.write(4, 2 + 12, format(miou, '.5'))

                workbook.save(args.log_dir + '/' + 'test' + '_xlwt.xls')

                log_string(LOG_FOUT, 'mean loss: %f' % (loss_sum / float(num_batches)))
                log_string(LOG_FOUT, 'perclass accuracy:')
                log_string(LOG_FOUT, str(per_class_accuracy))
                log_string(LOG_FOUT, 'total_accuracy: %f' % (mean_accuracy))
                log_string(LOG_FOUT, 'perclass iou:')
                log_string(LOG_FOUT, str(per_class_iou))
                log_string(LOG_FOUT, 'miou: %f' % (miou))
                log_string(LOG_FOUT, '')

            else:
                loss_value, out, Gt = sess.run([reduced_loss, pred_out, label_batch],
                                               feed_dict={epoch: Epoch, is_test: istest})

                total_accuracy += getaccuracy(out, Gt, VALIDCLASS)
                loss_sum += loss_value

    # train
    for epo in range(args.num_epochs):
        train_one_epoch(TRAIN_NUM_ONEEPOCH, args.save_pred_every, args.batch_size, epo)
        #eval_one_epoch(TEST_NUM_ONEEPOCH, args.save_pred_every, args.batch_size, epo)

    LOG_FOUT.close()

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
