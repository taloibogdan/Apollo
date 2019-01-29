import tensorflow as tf


def max_pool(input, **params):
    output, switches = tf.nn.max_pool_with_argmax(input, **params)

    with tf.variable_scope(params['name']):
        switches = tf.identity(switches,
                               name = 'switches')
        output = tf.identity(output,
                             name = 'output')

    return output


def avg_pool(input, **params):
    output = tf.layers.average_pooling2d(input,
                                       pool_size = [input.shape[1].value,
                                                    input.shape[2].value],
                                       strides = 100)
    output = tf.reshape(output, [-1, output.shape[-1].value])
    return output

