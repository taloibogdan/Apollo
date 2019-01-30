import tensorflow as tf


def input(input, **params):
    return input


def sigmoid(input, **params):
    output = tf.nn.sigmoid(input)
    return output


def reshape(input, **params):
    output = tf.reshape(input, params['shape'])
    return output


def to_tensor(input):
    if type(input) is str:
        return tf.get_default_graph().get_tensor_by_name(input)
    return input


def concat(input, **params):
    tensor_list = [input] + [to_tensor(t) for t in params['other']]
    return tf.concat(tensor_list, axis = params['axis'])


def flatten(input, **params):
    return tf.contrib.layers.flatten(input)


def reduce_max(input, **params):
    return tf.reduce_max(input, **params)


def argmax(input, **params):
    return tf.argmax(input, **params)


def dropout(input, **params):
    return tf.nn.dropout(input, **params)


def batch_norm(input, **params):
    return tf.layers.batch_normalization(
        input,
        training = to_tensor(params['is_training'])
    )


def fc(input, **params):
    with tf.variable_scope(params.get('name', '')):
        n_inputs = input.shape[-1].value
        n_outputs = params['units']
        initialization = params['initialization']

        kernel = tf.Variable(
            initial_value = initialization(shape = [n_inputs, n_outputs]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [n_outputs]),
            name = 'bias',
            dtype = tf.float32)

        weighted_input = tf.matmul(input, kernel,
                                   name = 'weighted_input')
        output_raw = tf.add(weighted_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


def fc_parallel(input, **params):  # for [n_batch, n_parallel, n_inputs]
    with tf.variable_scope(params.get('name', '')):
        n_outputs = params['units']
        initialization = params['initialization']
        n_inputs = input.shape[-1].value

        kernel = tf.Variable(
            initial_value = initialization(shape = [n_inputs, n_outputs]),
            name = 'kernel',
            dtype = tf.float32)
        kernel = tf.expand_dims(kernel, 0)
        kernel = tf.tile(kernel, [tf.shape(input)[0], 1, 1])

        bias = tf.Variable(
            initial_value = tf.zeros(shape = [n_outputs]),
            name = 'bias',
            dtype = tf.float32)

        weighted_input = tf.matmul(input, kernel,
                                   name = 'weighted_input')
        output_raw = tf.add(weighted_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


def scrambler(input, **params):
    with tf.variable_scope(params['name']):
        n_untouched = params['n_untouched']
        n_inputs = input.get_shape()[-1].value
        rands = tf.random_uniform([n_inputs-n_untouched], dtype=tf.float32)
        ones = tf.ones([n_untouched], dtype=tf.float32)
        mask = tf.concat([ones, rands], 0)
        output = input*mask
        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output, name='output')
        return output
