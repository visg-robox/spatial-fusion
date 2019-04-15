# -*- coding: utf-8 -*-

import tensorflow as tf
from data_process import data_loader_tf
import common

# Hyper Parameters
n_input = common.class_num # MNIST data 输入 (img shape: 28*28)
n_steps = 100 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = common.class_num  # MNIST 列别 (0-9 ，一共10类)

EPOCH = 10
learning_rate = 0.001
training_iters = 100000
batch_size = 512
display_step = 100


if __name__ == "__main__":
    # data process
    data_path = '/home/zhangjian/code/data/CARLA_episode_0019/test1/'
    infer_path = data_path + 'infer/'
    gt_path = data_path + 'gt/'

    # train data
    infer_hashmap, gt_hashmap,  keys_list = data_loader_tf.data_to_hashmap(infer_path, gt_path)

    # test data
    # test data location
    test_data_size = 10000
    test_infer_path = data_path + 'test/infer/'
    test_gt_path = data_path + 'test/gt/'

    # read data
    test_infer, test_gt, test_keys_list = data_loader_tf.data_to_hashmap(test_infer_path, test_gt_path)
    test_keys = test_keys_list[0 * test_data_size:(0 + 1) * test_data_size]

    # pre-process
    test_input = data_loader_tf.labelmap_to_batch(test_infer, test_keys, test_data_size, n_steps, n_input)
    test_gt = data_loader_tf.labelmap_to_gt_onehot(test_gt, test_keys, test_data_size, n_input)

    # tf Graph input
    x = tf.placeholder("float", [batch_size, n_steps, n_input])
    y = tf.placeholder("float", [batch_size, n_input])

    #
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
    mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
    x1 = tf.unstack(x, n_steps, 1)

    # LSTMCell
    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(mcell, x1, dtype=tf.float32)
    pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn = None)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 启动session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        for epoch in range(EPOCH):
            for i in range(len(keys_list) // batch_size):
                current_keys = keys_list[i * batch_size:(i + 1) * batch_size]
                batch_x = data_loader_tf.labelmap_to_batch\
                batch_x = data_loader_tf.labelmap_to_batch\
                    (infer_hashmap, current_keys, batch_size, n_steps, n_input)
                batch_y = data_loader_tf.labelmap_to_gt_onehot(gt_hashmap, current_keys, batch_size, n_input)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                if step % display_step == 0:
                    # 计算批次数据的准确率
                    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                step += 1
        print(" Finished!")



        # # 计算准确率 for 128 mnist test images
        # test_len = 128
        # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        # test_label = mnist.test.labels[:test_len]
        # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

