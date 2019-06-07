"""DeepLab v3 models based on slim library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import model_generator
import numpy as np
from utils import preprocessing
import dataset_util


#为了节省空间，确定tfrecord是否存储图片
_RECORD_IMG = False
_LOG_TRAIN_IMG_STEP = 100
_LOG_VAL_IMG_STEP = int(dataset_util.NUM_IMAGES['validation'] / 10)
BLANCE_WEIGHT = np.array([1,1,1,1,1,1,1,1,1,1,10,10,10,10,10,10,10,10,10,10],dtype = np.float)



#多GPU梯度结算
def average_gradients(tower_grads):
    average_grads = []

    for grad_and_vars in zip(*tower_grads):

        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

#模型函数
def model_fn(features, labels, mode, params):
    """Model function for PASCAL VOC."""
    with tf.name_scope('data_feed'):
        gpu_num = params['gpu_num'] if mode == tf.estimator.ModeKeys.TRAIN else 1
        next_img = features['image']
        input_shape = tf.shape(next_img)[1:3]
        if mode == tf.estimator.ModeKeys.PREDICT:
            next_img = tf.image.resize_bilinear(next_img,[dataset_util.HEIGHT,dataset_util.WIDTH])
        image_splits = tf.split(next_img, gpu_num, axis=0)

        if mode == tf.estimator.ModeKeys.TRAIN:
            batch_size = params['batch_size']
        else:
            batch_size = 1

    with tf.name_scope('model'):
        outdict = []
        network = model_generator(params['num_classes'],
                                     batch_norm_decay=params['batch_norm_decay'])
        for i in range(gpu_num):
            with tf.device('/gpu:%s' % i):
                if not params['freeze_batch_norm']:
                    out = network(image_splits[i], mode==tf.estimator.ModeKeys.TRAIN)
                else:
                    out = network(image_splits[i], is_training=False)
                outdict.append(out)

        out = outdict[0]
        logits_origin = out['logits']
        logits = tf.image.resize_bilinear(logits_origin, input_shape)
        train_var = out['train_var']

    with tf.name_scope('prediction'):
        pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

        pred_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                         [pred_classes, batch_size, params['num_classes']],
                                         tf.uint8)

        predictions = {}

        if mode == tf.estimator.ModeKeys.PREDICT:
            feature_out = out['features']
            probabilities_orgin = tf.nn.softmax(logits_origin, name='softmax_tensor')
            probabilities = tf.nn.softmax(logits, name='softmax_tensor')
            pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

            predictions = {
                'classes': pred_classes,
                'probabilities': probabilities,
                'probabilities_origin': probabilities_orgin,
                'feature_out': feature_out
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'preds': tf.estimator.export.PredictOutput(predictions)
                })

        labels = tf.squeeze(labels, axis=3)
        label_splits = tf.split(labels, gpu_num, axis=0)
        # reduce the channel dimension.
        gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                       [tf.expand_dims(label_splits[0], axis=3), batch_size, params['num_classes']],
                                       tf.uint8)


    def get_cross_entropy(logits,labels):
        logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
        labels_flat = tf.reshape(labels, [-1, ])

        valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
        valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
        valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=valid_logits, labels=valid_labels)
        return cross_entropy




        # Add weight decay to the loss.

    #训练设置和梯度反传
    with tf.name_scope('train_config'):
        global_step = tf.train.get_or_create_global_step()
        if not params['freeze_batch_norm']:
            train_var_list = [v for v in train_var]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        else:
            update_ops = None
            train_var_list = [v for v in train_var
                              if 'beta' not in v.name and 'gamma' not in v.name]

        if mode == tf.estimator.ModeKeys.TRAIN:
            if params['learning_rate_policy'] == 'piecewise':
                # Scale the learning rate linearly with the batch size. When the batch size
                # is 128, the learning rate should be 0.1.
                initial_learning_rate = 0.1 * params['batch_size'] / 128
                batches_per_epoch = params['num_train'] / params['batch_size']
                # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
                boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
                values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
                learning_rate = tf.train.piecewise_constant(
                    tf.cast(global_step, tf.int32), boundaries, values)

            elif params['learning_rate_policy'] == 'warm':
                learning_rate = tf.constant(1e-4)

            elif params['learning_rate_policy'] == 'poly':
                learning_rate = tf.train.polynomial_decay(
                    params['initial_learning_rate'],
                    tf.cast(global_step, tf.int32) - params['initial_global_step'],
                    params['max_iter'], params['end_learning_rate'], power=params['power'],cycle=False)
            elif params['learning_rate_policy'] == 'exponential':
                learning_rate = tf.train.exponential_decay(
                    params['initial_learning_rate'],  # Base learning rate.
                    tf.cast(global_step, tf.int32) - params['initial_global_step'],  # Current index into the dataset.
                    params['num_train'],  # Decay step.
                    params['decay_rate'],  # Decay rate.
                    staircase=True)
                learning_rate = tf.maximum(learning_rate, params['end_learning_rate'])  # CLIP THE LEARNING RATE!!
            else:
                raise ValueError('Learning rate policy must be "piecewise" or "poly"')

            #lr for warm up choice
            learning_rate = tf.where(global_step < params['warm_up_step'], params['warm_up_lr'], learning_rate)
            tf.identity(learning_rate, name='learning_rate')
            tf.summary.scalar('learning_rate', learning_rate)
            # Create a tensor named learning_rate for logging purposes

            if params['optimizer'] == 'M':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=params['momentum'])

            if params['optimizer'] == 'A':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


            #multi-gpu gradient calculate

            tower_grads = []
            tower_CE = []
            with tf.name_scope("total_loss"):
                for i in range(gpu_num):
                    with tf.device('/gpu:%s' % i):
                        logit = tf.image.resize_bilinear(outdict[i]['logits'], input_shape)
                        ce = get_cross_entropy(logit, label_splits[i])
                        loss =  ce + params[
                            'weight_decay'] * tf.add_n(
                            [tf.nn.l2_loss(v) for v in train_var_list])
                        grads = optimizer.compute_gradients(loss, var_list=train_var_list)
                        tower_grads.append(grads)
                        tower_CE.append(ce)

            grads = average_gradients(tower_grads)
            cross_entropy = tf.reduce_mean(tower_CE)


            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(grads, global_step=global_step)

            writer_dir=params['model_dir']+'/my_eval/train'
            log_step = _LOG_TRAIN_IMG_STEP

        else:
            log_step = _LOG_VAL_IMG_STEP
            loss = tf.constant(1)
            train_op = None
            writer_dir = params['model_dir'] + '/my_eval/test'

    #记录图像和准确率
    with tf.name_scope('single_loss_for_log_and_eval'):
        with tf.device('/gpu: 0'):
            labels_flat = tf.reshape(label_splits[0], [-1, ])
            valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
            valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
            preds_flat = tf.reshape(pred_classes, [-1, ])
            valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
            confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

    with tf.name_scope('summary'):
        #记录图像
        eval_hook = []
        images = image_splits[0]
        images = tf.cast(
            tf.map_fn(preprocessing.mean_image_addition, images),
            tf.uint8)
        visiolize= tf.concat(values=[images ,gt_decoded_labels,pred_decoded_labels], axis=1)
        temp_shape= images.get_shape().as_list()
        print(temp_shape)
        visionlize = tf.slice(visiolize, [0, 0, 0, 0], [1, temp_shape[1]*3, temp_shape[2], temp_shape[3]])
        img_summary = tf.summary.image('images', visionlize,
                                       max_outputs=params['tensorboard_images_max_outputs'])

        if(_RECORD_IMG):
            eval_hook.append(tf.train.SummarySaverHook(summary_op=img_summary, output_dir=writer_dir, save_steps=log_step))

        # Create a tensor named train_accuracy for logging purposes
        # Create a tensor named cross_entropy for logging purposes.
        accuracy = tf.metrics.accuracy(
            valid_labels, valid_preds)
        mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
        metrics = {'val_px_accuracy': accuracy, 'val_mean_iou': mean_iou}

        #准确率
        tf.identity(accuracy[1], name='train_px_accuracy')
        tf.summary.scalar('train_px_accuracy', accuracy[1])

        #miou
        def compute_mean_iou(total_cm, classname,name='mean_iou'):
            """Compute the mean intersection-over-union via the confusion matrix."""
            sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
            sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
            cm_diag = tf.to_float(tf.diag_part(total_cm))
            denominator = sum_over_row + sum_over_col - cm_diag

            # The mean is only computed over classes that appear in the
            # label or prediction tensor. If the denominator is 0, we need to
            # ignore the class.
            num_valid_entries = tf.reduce_sum(tf.cast(
                tf.not_equal(denominator, 0), dtype=tf.float32))

            # If the value of the denominator is 0, set it to 1 to avoid
            # zero division.
            denominator = tf.where(
                tf.greater(denominator, 0),
                denominator,
                tf.ones_like(denominator))
            iou = tf.div(cm_diag, denominator)

            for i in range(params['num_classes']):
                tf.identity(iou[i], name='train_iou_class{}'.format(i))
                iou_summary=tf.summary.scalar('train_iou_class{}_'.format(i)+classname[i], iou[i])
                eval_hook.append(tf.train.SummarySaverHook(summary_op=iou_summary,output_dir=writer_dir, save_secs=2))



            # If the number of valid entries is 0 (no classes) we return 0.
            result = tf.where(
                tf.greater(num_valid_entries, 0),
                tf.reduce_sum(iou, name=name) / num_valid_entries,
                0)
            return result
        train_mean_iou = compute_mean_iou(mean_iou[1],classname=params['classname'])
        tf.identity(train_mean_iou, name='train_mean_iou')
        tf.summary.scalar('train_mean_iou', train_mean_iou)


    #prediction for evaluate
    with tf.name_scope('prediction'):
        predictions['visionlize'] = visionlize
        predictions['valid_preds'] = valid_preds
        predictions['valid_labels'] = valid_labels
        predictions['confusion_matrix'] = confusion_matrix

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        evaluation_hooks=eval_hook,
        eval_metric_ops=metrics)
