from __future__ import division
import tensorflow as tf
from models import *
import numpy as np

FLAGS = tf.app.flags.FLAGS

# import cifar10 data
from tensorflow.models.image.cifar10 import cifar10
cifar10.maybe_download_and_extract()

# global variable to select which (and how many) GPU's to use
# (tensorflow can be hungry with resources if not properly controlled)
gpus_to_use = [3]

# network input (data and correct labels)
# x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])

train_images, train_labels = cifar10.distorted_inputs()
test_images, test_labels = cifar10.inputs(eval_data=True)

# select stream to use (train or test)
select_test = tf.placeholder(dtype=bool,shape=[],name='select_test')
x = tf.cond(
    select_test,
    lambda:test_images,
    lambda:train_images
)
y_ = tf.cond(
    select_test,
    lambda:test_labels,
    lambda:train_labels
)

# first convolutional layer
with tf.variable_scope("conv1"):
    h_conv1 = conv2D(x, [5, 5, 3, 32], [32])
    h_pool1 = max_pool(h_conv1, 2)

# second convolutional layer
with tf.variable_scope("conv2"):
    h_conv2 = conv2D(h_pool1, [5, 5, 32, 64], [64])
    h_pool2 = max_pool(h_conv2, 2)

# fully connected layers

with tf.variable_scope("fc1"):

    # reshape output of pooling into many flat vectors
    reshape = tf.reshape(h_pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = fully_connected(reshape, dim, 1024)

# dropout layer (with variable dropout rate)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.variable_scope("fc2"):
    y_conv = fully_connected(h_fc1_drop, 1024, 10)

# calculate cross entropy on softmax (logits refers to the use of logs
# on calclulations, meaning those are stable calcs)
y_ = tf.cast(y_, tf.int64)
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_))

# setup an Adam optimizer to minimize the just-calculated cross entropy
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)

# add an extra end node to the graph representing accuracy
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
top_k_op = tf.nn.in_top_k(y_conv, y_, 1)

# start tensorflow session
c = tf.ConfigProto()
c.gpu_options.visible_device_list=','.join(map(str, gpus_to_use))
sess = tf.Session(config=c)
sess.run(tf.initialize_all_variables())

with sess.as_default():

    # header of csv style output format
    print("epoch, training error, training accuracy")
    tf.train.start_queue_runners()

    # run for 20k epochs
    for i in range(500):

        # check accuracy
        if i%10 == 0:
            train_acurracy = np.sum(top_k_op.eval(feed_dict={keep_prob: 1.0, select_test: False}))/FLAGS.batch_size
            train_error = cross_entropy.eval(feed_dict={keep_prob: 1.0, select_test: False})

            print("%d, %g, %g"%(i, train_error, train_acurracy))

        # run optimizer on batch
        train_step.run(feed_dict={keep_prob: 0.5, select_test: False})

    print ("test error, test accuracy")
    # final accuracy (test)
    for i in range(100):
        acc, error = sess.run([top_k_op, cross_entropy], feed_dict={keep_prob: 1.0, select_test: True})
        acc = np.sum(acc)/FLAGS.batch_size
        # acc = np.sum(top_k_op.run(feed_dict={keep_prob: 1.0, select_test: True}))/FLAGS.batch_size
        # error = cross_entropy.run(feed_dict={keep_prob: 1.0, select_test: True})
        print("%g, %g"%(error, acc))