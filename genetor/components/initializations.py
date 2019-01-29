import tensorflow as tf


def xavier_over_2(shape, variable_type):
    if variable_type == 'fc_kernel':
        return tf.random_normal(shape = shape,
                                dtype = tf.float32) / tf.sqrt(float(shape[0]) / 2)

    if variable_type == 'fc_bias':
        return tf.random_normal(shape = shape,
                                dtype = tf.float32,
                                stddev = 0.01)

    return None


def default_initialization(shape):
    return tf.random_normal(shape = shape,
                            mean = 0.0,
                            stddev = 0.05,
                            dtype = tf.float32)

