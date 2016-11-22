import tensorflow as tf

bias_const_init=0.1
weight_stddev_init=0.1

def conv2D(x, kernel_shape, bias_shape):

    # weight and bias
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=weight_stddev_init))
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(bias_const_init))

    # conv operation and activation
    conv_op = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
    activation = tf.nn.relu(conv_op + biases)
    return activation

def fully_connected(x, input_size, output_size):
    weights = tf.get_variable("weights", [input_size, output_size], initializer=tf.truncated_normal_initializer(stddev=weight_stddev_init))
    biases = tf.get_variable("biases", output_size,
        initializer=tf.constant_initializer(bias_const_init))
    activation = tf.nn.relu(tf.matmul(x, weights) + biases)
    return activation

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')