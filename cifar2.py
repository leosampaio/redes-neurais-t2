from __future__ import division
import tensorflow as tf
from models import *
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('optimizer', 'adam',
                            """Optimizer of choice""")
tf.app.flags.DEFINE_integer('gpu', 1,
                            """GPU to run model on""")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/adam/',
                            """folder to save or load checkpoint""")
tf.app.flags.DEFINE_boolean('evaluate', False,
                            """run in evaluation mode""")

batch_size = 128
nb_classes = 10
nb_epoch = 100000

# image info
img_rows, img_cols = 32, 32
img_channels = 3

# global variable to select which (and how many) GPU's to use
# (tensorflow can be hungry with resources if not properly controlled)
gpus_to_use = [FLAGS.gpu]

# network input (data and correct labels)
x = tf.placeholder(tf.float32, shape=[None, img_rows, img_cols, img_channels])
y_ = tf.placeholder(tf.float32, shape=[None, nb_classes])

# first convolutional layer
with tf.variable_scope("conv1"):
    h_conv1 = conv2D(x, [5, 5, img_channels, 32], [64])
    h_pool1 = max_pool(h_conv1, 2)

# norm1
norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

# second convolutional layer
with tf.variable_scope("conv2"):
    h_conv2 = conv2D(norm1, [3, 3, 32, 64], [64])

# norm2
norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')

# third convolutional layer
with tf.variable_scope("conv3"):
    h_conv3 = conv2D(norm2, [3, 3, 64, 64], [64])
    h_pool2 = max_pool(h_conv3, 2)

norm3 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')

# fully connected layers

# reshape output of pooling into many flat vectors
h_pool2_flat = tf.reshape(norm3, [-1, 8*8*64])

with tf.variable_scope("fc1"):
    h_fc1 = fully_connected(h_pool2_flat, 8*8*64, 512)

# dropout layer (with variable dropout rate)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.variable_scope("fc2"):
    y_conv = fully_connected(h_fc1_drop, 512, nb_classes)

# calculate cross entropy on softmax (logits refers to the use of logs
# on calclulations, meaning those are stable calcs)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# setup an optimizer to minimize the just-calculated cross entropy
if (FLAGS.optimizer == 'adam'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
elif (FLAGS.optimizer == 'adagrad'):
  train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
elif (FLAGS.optimizer == 'sgd'):
  train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)

# add an extra end node to the graph representing accuracy
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# start tensorflow session
c = tf.ConfigProto()
c.gpu_options.visible_device_list=','.join(map(str, gpus_to_use))
sess = tf.Session(config=c)
sess.run(tf.initialize_all_variables())

# prepare data

# import cifar10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)
generator = datagen.flow(X_train, Y_train, batch_size=batch_size)

with sess.as_default():

    if not FLAGS.evaluate:
        # header of csv style output format
        print("epoch, training error, training accuracy")

        # run for 20k epochs
        for i in range(nb_epoch):
            # with batch size of 50
            images, labels = generator.next()

            # check accuracy
            if i%50 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:images, y_: labels, keep_prob: 1.0})
                train_error = cross_entropy.eval(feed_dict={
                    x:images, y_: labels, keep_prob: 1.0})
                print("%d, %g, %g"%(i, train_error, train_accuracy))

            # save checkpoint for loading later
            if i%1000 == 0:
                if not os.path.exists(FLAGS.checkpoint_dir):
                    os.makedirs(FLAGS.checkpoint_dir)
                path = FLAGS.checkpoint_dir + "model.ckpt"
                save_path = saver.save(sess, path)


            # run optimizer on batch
            sess.run([train_step], feed_dict={x: images, y_: labels, keep_prob: 0.5})
    else:
        path = FLAGS.checkpoint_dir + "model.ckpt"
        saver.restore(sess, path)
        # final accuracy
        print("Final Test Accuracy is %g"%accuracy.eval(feed_dict={
            x: X_test, y_: Y_test, keep_prob: 1.0}))