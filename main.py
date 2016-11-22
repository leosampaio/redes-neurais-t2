import tensorflow as tf
from models import *

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# global variable to select which (and how many) GPU's to use
# (tensorflow can be hungry with resources if not properly controlled)
gpus_to_use = [1]

# network input (data and correct labels)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# reshape mnist back to 28x28 to apply convolution
x_image = tf.reshape(x, [-1,28,28,1])

# first convolutional layer
with tf.variable_scope("conv1"):
    h_conv1 = conv2D(x_image, [5, 5, 1, 32], [32])
    h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
with tf.variable_scope("conv2"):
    h_conv2 = conv2D(h_pool1, [5, 5, 32, 64], [64])
    h_pool2 = max_pool_2x2(h_conv2)

# fully connected layers

# reshape output of pooling into many flat vectors
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

with tf.variable_scope("fc1"):
    h_fc1 = fully_connected(h_pool2_flat, 7*7*64, 1024)

# dropout layer (with variable dropout rate)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.variable_scope("fc2"):
    y_conv = fully_connected(h_fc1_drop, 1024, 10)

# calculate cross entropy on softmax (logits refers to the use of logs
# on calclulations, meaning those are stable calcs)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# setup an Adam optimizer to minimize the just-calculated cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# add an extra end node to the graph representing accuracy
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start tensorflow session
c = tf.ConfigProto()
c.gpu_options.visible_device_list=','.join(map(str, gpus_to_use))
sess = tf.Session(config=c)
sess.run(tf.initialize_all_variables())

with sess.as_default():

    # run for 20k epochs
    for i in range(20000):
        # with batch size of 50
        batch = mnist.train.next_batch(50)

        # check accuracy
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

        # run optimizer on batch
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # final accuracy
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))