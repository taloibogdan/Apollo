import tensorflow as tf
from .basic import to_tensor
from .initializations import default_initialization
from .default_params import DEFAULT_PARAMS


def conv(input, **params):
    with tf.variable_scope(params['name']):
        kernel_size = params['kernel_size']
        filters = params['filters']
        strides = [1, params['stride'], params['stride'], 1]
        padding = params['padding']
        initialization = params['initialization']

        kernel = tf.Variable(
            initial_value = initialization(
                shape = [kernel_size, kernel_size, input.shape[-1].value, filters]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [filters]),
            name = 'bias',
            dtype = tf.float32)

        filtered_input = tf.nn.conv2d(
            input = input,
            filter = kernel,
            strides = strides,
            padding = padding,
            name = 'filtered_input')
        output_raw = tf.add(filtered_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


def dilated_conv(input, **params):
    with tf.variable_scope(params['name']):
        kernel_size = params['kernel_size']
        filters = params['filters']
        rate = params['rate']
        padding = params['padding']
        initialization = params['initialization']

        kernel = tf.Variable(
            initial_value = initialization(
                shape = [kernel_size, kernel_size, input.shape[-1].value, filters]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [filters]),
            name = 'bias',
            dtype = tf.float32)

        filtered_input = tf.nn.atrous_conv2d(
            value = input,
            filters = kernel,
            rate = rate,
            padding = padding,
            name = 'filtered_input')
        output_raw = tf.add(filtered_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


def skip(input, **params):
    with tf.variable_scope(params['name']):
        other = to_tensor(params['other'])

        if input.shape[1].value == other.shape[1].value:
            output = tf.add(input, other, name = 'output')
        else:
            conv_other = conv(other,
                              name = 'skip_conv',
                              filters = input.shape[-1].value,
                              kernel_size = 1, stride = 2,
                              padding = 'SAME', activation = tf.nn.relu,
                              initialization = default_initialization)

            output = tf.add(input, conv_other, name = 'output')

    return output


def fire(input, **params):
    with tf.variable_scope(params['name']):
        params['squeeze'] = params.get('squeeze', {})
        params['squeeze'] = dict(DEFAULT_PARAMS['conv'],
                                 **params['squeeze'],
                                 kernel_size = 1,
                                 name = 'squeeze')
        squeezed_input = conv(input,
                              **params['squeeze'])

        params['expand_obo'] = params.get('expand_obo', {})
        params['expand_obo'] = dict(DEFAULT_PARAMS['conv'],
                                    **params['expand_obo'],
                                    kernel_size = 1,
                                    name = 'expand_obo')
        obo_conv = conv(squeezed_input,
                        **params['expand_obo'])

        params['expand_tbt'] = params.get('expand_tbt', {})
        params['expand_tbt'] = dict(DEFAULT_PARAMS['conv'],
                                    **params['expand_tbt'],
                                    kernel_size = 3,
                                    name = 'expand_tbt')
        tbt_conv = conv(squeezed_input,
                        **params['expand_tbt'])

        output = tf.concat([obo_conv, tbt_conv],
                           axis = -1,
                           name = 'output')

    return output


def bypass_fire(input, **params):
    with tf.variable_scope(params['name']):
        fire_output = fire(input, **dict(params, name = 'fire'))

        output = tf.add(input, fire_output,
                         name = 'output')

    return output

