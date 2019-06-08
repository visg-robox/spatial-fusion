"""Train a DeepLab v3 plus model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import dataset_util
from config import model_fn
from utils import preprocessing
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
import timeit
import shutil



#数据增强策略
_MIN_SCALE = 0.8
_MAX_SCALE = 1.5

#使用几块GPU训练和输出分辨率为多大
_GPU_NUM = 1
_BATCH_SIZE = 10
_BUFFER_SIZE = 60
_CROP_HEIGHT = 1024
_CROP_WIDTH = 1024

#训练主要超参设置
_MAX_ITER = 15000
_MIDDLE_STEP = 0
_EPOCH = (_MAX_ITER - _MIDDLE_STEP)//(dataset_util.NUM_IMAGES['train']//_BATCH_SIZE) + 2
print(_EPOCH)
_INITIAL_LR = 5e-3
_INITIAL_STEP = 0
_END_LR = 1e-6
_WARM_UP_LR = 1e-4
_WARM_UP_STEP = 2000
_WEIGHT_DECAY = 5e-5

#优化策略选择
_OPTIMIZER = 'M'        # 'M' for momentum,'A' for Adam
_POWER = 0.9
_MOMENTUM = 0.9
_DECAY_POLICY = 'poly'  #choices=['poly', 'piecewise','exponential' ,'warm']
_EX_LR_DECAY_RATE=0.8

#BN层设置
_FREEZE_BN = False
_BATCH_NORM_DECAY = 0.997

MODEL_DIR  = 'ICNET_30000_balance_new'
MODEL_DIR = os.path.join('../data_and_checkpoint', dataset_util.DATASET_SHOT, 'model_checkpoint', MODEL_DIR)

print(MODEL_DIR)



parser = argparse.ArgumentParser()


parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

parser.add_argument('--optimizer', type=str, default=_OPTIMIZER,
                    choices=['A', 'M' ],
                    help='use Adam or momentum optimizer')

parser.add_argument('--data_dir', type=str, default = dataset_util.DATA_DIR,
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--model', type=str, choice = ['I','D'],
                    help='ICNET or Deeplab.')

parser.add_argument('--pretrained_model', type=str,
                    help='pretrained_model')

parser.add_argument('--epochs_per_eval', type=int, default=2,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=4,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--output_stride', type=int, default=16,
                    help='model_output_stride')

parser.add_argument('--batch_size', type=int, default=_BATCH_SIZE,
                    help='Number of examples per batch.')

parser.add_argument('--train_epochs', type=int, default=_EPOCH,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). ')

parser.add_argument('--max_iter', type=int, default=_MAX_ITER,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--learning_rate_policy', type=str, default= _DECAY_POLICY,
                    choices=['poly', 'piecewise','exponential' ,'warm'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--initial_learning_rate', type=float, default=_INITIAL_LR,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=_END_LR,
                    help='End learning rate for the optimizer.')

parser.add_argument('--warm_up_learning_rate', type=float, default=_WARM_UP_LR,
                    help='End learning rate for the optimizer.')

parser.add_argument('--warm_up_step', type=float, default=_WARM_UP_STEP,
                    help='End learning rate for the optimizer.')

parser.add_argument('--weight_decay', type=float, default=_WEIGHT_DECAY,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--bn_decay', type=float, default=_BATCH_NORM_DECAY,
                    help='The batch norm decay.')

parser.add_argument('--initial_global_step', type=int, default=_INITIAL_STEP,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--model_dir', type=str, default=MODEL_DIR,
                    help='Base directory for the model.')

parser.add_argument('--freeze_batch_norm', default=_FREEZE_BN,
                    help='Freeze batch normalization parameters during the training.')




def get_filenames(is_training, data_dir):
    """Return a list of filenames.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: path to the the directory containing the input data.

    Returns:
      A list of file names.
    """
    if is_training:
        return [os.path.join(data_dir, 'train.tfrecords')]
    else:
        return [os.path.join(data_dir, 'val.tfrecords')]

def preprocess_image(image, label, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Randomly scale the image and label.
        image, label = preprocessing.random_rescale_image_and_label(
            image, label, _MIN_SCALE, _MAX_SCALE)
        # #
        # # # # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        image,  label = preprocessing.random_crop_or_pad_image_and_label(
            image, label, _CROP_HEIGHT, _CROP_WIDTH, dataset_util.IGNORE_LABEL)

        # Randomly flip the image and label horizontally.
        image, label = preprocessing.random_flip_left_right_image_and_label(
            image, label)

    image = preprocessing.mean_image_subtraction(image)

    return image, label

def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      A tuple of images and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=_BUFFER_SIZE)

    dataset = dataset.map(dataset_util.parse_record)
    dataset = dataset.map(
        lambda image, label: preprocess_image(image, label, is_training))
    dataset = dataset.prefetch(batch_size)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    features={}
    features['image']=images
    return features, labels





def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    if FLAGS.clean_model_dir:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

    tmp_path = os.path.join(FLAGS.model_dir, 'pyfile')
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)

    shutil.copyfile('model.py', os.path.join(tmp_path, 'model.py'))
    shutil.copyfile('config.py', os.path.join(tmp_path, 'config.py'))
    shutil.copyfile('train.py', os.path.join(tmp_path, 'train.py'))
    shutil.copyfile('dataset_util.py', os.path.join(tmp_path, 'dataset_util.py'))
    Model_fn=model_fn

    # Set up a RunConfig to only save checkpoints once per training cycle.

    run_config = tf.estimator.RunConfig(session_config=tf.ConfigProto(allow_soft_placement=True)).replace(save_checkpoints_secs=1e9, keep_checkpoint_max=4)
    model = tf.estimator.Estimator(
        model_fn=Model_fn,
        model_dir=FLAGS.model_dir,
        config=run_config,
        params={
            'model_dir': FLAGS.model_dir,
            'output_stride': FLAGS.output_stride,
            'batch_size': FLAGS.batch_size,
            'batch_norm_decay': FLAGS.bn_decay,
            'num_classes': dataset_util.NUM_CLASSES,
            'classname':dataset_util.CLASSNAME,
            'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
            'weight_decay': FLAGS.weight_decay,
            'learning_rate_policy': FLAGS.learning_rate_policy,
            'num_train': dataset_util.NUM_IMAGES['train'],
            'initial_learning_rate': FLAGS.initial_learning_rate,
            'max_iter': FLAGS.max_iter,
            'end_learning_rate': FLAGS.end_learning_rate,
            'power': _POWER,
            'momentum': _MOMENTUM,
            'optimizer':FLAGS.optimizer,
            'freeze_batch_norm': FLAGS.freeze_batch_norm,
            'initial_global_step': FLAGS.initial_global_step,
            'decay_rate':_EX_LR_DECAY_RATE,
            'warm_up_lr':FLAGS.warm_up_learning_rate,
            'warm_up_step':FLAGS.warm_up_step,
            'gpu_num': _GPU_NUM,
            'model': FLAGS.model,
            'pretrained_model': FLAGS.pretrained_model

        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'train_config/learning_rate',
            'train_px_accuracy': 'summary/train_px_accuracy',
            'train_mean_iou': 'summary/train_mean_iou',
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)
        train_hooks = [logging_hook]

        tensors_to_log = {
            'train_px_accuracy': 'summary/train_px_accuracy',
        }


        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)
        eval_hooks = [logging_hook]


        if FLAGS.debug:
            debug_hook = tf_debug.LocalCLIDebugHook()
            train_hooks.append(debug_hook)
            eval_hooks = [debug_hook]

   
        if 1:
            tf.logging.info("Start training.")
            model.train(
                input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
                hooks=train_hooks,
                # steps=1  # For debug
            )

        if 1:
            tf.logging.info("Start evaluation.")
            # Evaluate the model and print results
            eval_results = model.evaluate(
                # Batch size must be 1 for testing because the images' size differs
                input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
                hooks=eval_hooks,
                # steps=1  # For debug
            )
            print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
