"""Evaluate a DeepLab v3 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.python.client import timeline
import config
import dataset_util

from utils import eval_API
import numpy as np
import timeit
from scipy import misc
import os
from train import MODEL_DIR,input_fn


parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', type=str, default=dataset_util.DATA_DIR,
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--model_dir', type=str, default=MODEL_DIR,
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--output_stride', type=int, default=16,
                    help='model_output_stride')


_IGNORE_LABEL=255
VAL_NUM=dataset_util.NUM_IMAGES['validation']
SAVE_DIR = MODEL_DIR +'/my_eval/visionlize'


def compute_mean_iou(total_cm):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    ious = cm_diag / denominator

    print('Intersection over Union for each class:')
    for i, iou in enumerate(ious):
        print('    class {}: {:.4f}'.format(i, iou))

    # If the number of valid entries is 0 (no classes) we return 0.
    m_iou = np.where(
        num_valid_entries > 0,
        np.sum(ious) / num_valid_entries,
        0)
    m_iou = float(m_iou)
    print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))

    print(sum_over_row.shape)
    total_acc = np.concatenate([np.expand_dims(sum_over_row, axis=1), np.expand_dims(cm_diag, axis=1),
                                np.expand_dims(sum_over_col, axis=1)], axis=1)
    print(total_acc.shape)
    eval_API.eval_print_save(total_acc, FLAGS.model_dir.split('/')[-1], FLAGS.model_dir + '/my_eval', np.array(dataset_util.CLASSNAME))


def main(unused_argv):
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    features, labels = input_fn(False, FLAGS.data_dir, 1)
    predictions = config.model_fn(
        features,
        labels,
        tf.estimator.ModeKeys.EVAL,
        params={
            'output_stride': FLAGS.output_stride,
            'batch_size': 1,  # Batch size must be 1 because the images' size may differ
            'pre_trained_model': None,
            'batch_norm_decay' : 1,
            'weight_decay':1e-4,
            'num_classes': dataset_util.NUM_CLASSES,
            'freeze_batch_norm': True,
            'classname': dataset_util.CLASSNAME,
            'model_dir': FLAGS.model_dir,
            'tensorboard_images_max_outputs':1,
            'gpu_num' : 0
        }).predictions

    # Manually load the latest checkpoint
    saver = tf.train.Saver()
    # run_metadata = tf.RunMetadata()
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # cf = tf.ConfigProto(graph_options=tf.GraphOptions(
    #     optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        #Loop through the batches and store predictions and labels

        #array for compute miou
        sum_cm = np.zeros((dataset_util.NUM_CLASSES, dataset_util.NUM_CLASSES), dtype=np.int32)

        for step in range(VAL_NUM):
            if not step%10:
                print('process', step, '/', VAL_NUM)
            # preds = sess.run(predictions, options=run_options, run_metadata=run_metadata)
            preds = sess.run(predictions)
            #record timeline
            # if step == 9:
            #     with open(FLAGS.model_dir + '/timeline2.json', 'w') as wd:
            #         tl = timeline.Timeline(run_metadata.step_stats)
            #         ctf = tl.generate_chrome_trace_format()
            #         wd.write(ctf)

            sum_cm += preds['confusion_matrix']
            visionlize = np.squeeze(preds['visionlize'])
            misc.imsave(SAVE_DIR + '/' + str(step) +'.png',np.array(visionlize,dtype = np.uint8))

        compute_mean_iou(sum_cm)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
