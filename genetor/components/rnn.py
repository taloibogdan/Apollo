import tensorflow as tf
from .basic import to_tensor


def make_RNN_cell_and_initial_state(initial_state, hidden_dims, cell_type, batch_size):
    cells_dict = {'LSTM': tf.nn.rnn_cell.LSTMCell, 'GRU': tf.nn.rnn_cell.GRUCell}
    cells = [cells_dict[cell_type](dim) for dim in hidden_dims]
    if len(cells) > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    else:
        cell = cells[0]

    state = None
    if initial_state is not None:
        if cell_type == 'LSTM':
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_state[0], initial_state[1])
        if len(hidden_dims) > 1:
            state = list(cell.zero_state(batch_size, tf.float32))
            state[0] = initial_state
            state = tuple(state)
        else:
            state = initial_state
    return cell, state


def reshape_RNN_last_state(cell_type, out_dim, states):
    if cell_type == 'LSTM':
        old_h = tf.concat(axis=1, values=[state.h for state in states])
        old_c = tf.concat(axis=1, values=[state.c for state in states])
        init_dim_h = old_h.shape[1]
        init_dim_c = old_c.shape[1]
        if init_dim_h == out_dim:
            new_h = old_h
        else:
            w_h = tf.get_variable('wh', [init_dim_h, out_dim], dtype=tf.float32)
            b_h = tf.get_variable('bh', [out_dim], dtype=tf.float32)
            new_h = tf.nn.relu(tf.matmul(old_h, w_h) + b_h)
        if init_dim_c == out_dim:
            new_c = old_c
        else:
            w_c = tf.get_variable('wc', [init_dim_c, out_dim], dtype=tf.float32)
            b_c = tf.get_variable('bc', [out_dim], dtype=tf.float32)
            new_c = tf.nn.relu(tf.matmul(old_c, w_c) + b_c)
        return tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
    else:
        old_h = tf.concat(axis=1, values=states)
        init_dim_h = old_h.shape[1]
        if init_dim_h == out_dim:
            new_h = old_h
        else:
            w_h = tf.get_variable('wh', [init_dim_h, out_dim], dtype=tf.float32)
            b_h = tf.get_variable('bh', [out_dim], dtype=tf.float32)
            new_h = tf.nn.relu(tf.matmul(old_h, w_h) + b_h)
        return new_h


def RNN(input, **params):
    with tf.variable_scope(params['name']):
        sequence_length = params['sequence_length']
        cell_type = params['cell_type']

        hidden_dims = params['hidden_dims']
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]

        batch_size = tf.shape(input)[0]

        initial_state = to_tensor(params['initial_state'])
        cell, initial_state = make_RNN_cell_and_initial_state(initial_state, hidden_dims, cell_type, batch_size)
        last_state_dim = params.get('last_state_dim', hidden_dims[-1])

        outputs, state = tf.nn.dynamic_rnn(cell, input,
                                           initial_state=initial_state,
                                           dtype=tf.float32,
                                           sequence_length=sequence_length,
                                           swap_memory=True)

        tf.identity(outputs, name='outputs')
        if len(hidden_dims) > 1:
            state = state[-1]
        state = reshape_RNN_last_state(cell_type, last_state_dim, [state])
        tf.identity(state, name='last_state')
        tf.identity(state[-1], name='last_state')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(outputs[:, -1, ], name='output')
        return output


def bidirectional_RNN(input, **params):
    with tf.variable_scope(params['name']):
        sequence_length = params['sequence_length']
        cell_type = params['cell_type']

        hidden_dims_fw = params.get('hidden_dims_fw', params['hidden_dims'])
        hidden_dims_bw = params.get('hidden_dims_bw', params['hidden_dims'])
        if not isinstance(hidden_dims_fw, list):
            hidden_dims_fw = [hidden_dims_fw]
        if not isinstance(hidden_dims_bw, list):
            hidden_dims_bw = [hidden_dims_bw]

        batch_size = tf.shape(input)[0]

        initial_state_fw = to_tensor(params['initial_state_fw'])
        initial_state_bw = to_tensor(params['initial_state_bw'])
        cell_fw, initial_state_fw = make_RNN_cell_and_initial_state(initial_state_fw, hidden_dims_fw,
                                                                    cell_type, batch_size)
        cell_bw, initial_state_bw = make_RNN_cell_and_initial_state(initial_state_bw, hidden_dims_bw,
                                                                    cell_type, batch_size)

        last_state_dim = params.get('last_state_dim', hidden_dims_fw[-1])

        outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input,
                                                                        initial_state_fw=initial_state_fw,
                                                                        initial_state_bw=initial_state_bw,
                                                                        dtype=tf.float32,
                                                                        sequence_length=sequence_length,
                                                                        swap_memory=True)

        if len(hidden_dims_fw) > 1:
            state_fw = state_fw[-1]
        if len(hidden_dims_bw) > 1:
            state_bw = state_bw[-1]
        state = reshape_RNN_last_state(cell_type, last_state_dim, [state_fw, state_bw])
        tf.identity(state, name='last_state')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        outputs = tf.concat(outputs, 2, name='outputs')
        output = activation(outputs[:, -1], name='output')
        return output


def attention_decoder(input, **params):
    with tf.variable_scope(params['name']):
        cell_type = params['cell_type']
        encoder_states = to_tensor(params['encoder_states'])  # [batch_size, attention_len, attention_size]
        encoder_mask = params['encoder_mask']
        hidden_dims = params['hidden_dims']
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]

        batch_size = tf.shape(encoder_states)[0]

        initial_state = to_tensor(params['initial_state'])
        cell, initial_state = make_RNN_cell_and_initial_state(initial_state, hidden_dims, cell_type, batch_size)

        attention_len = tf.shape(encoder_states)[1]
        attention_size = encoder_states.get_shape()[2]
        n_decoder_cells = input.get_shape()[1].value
        decoder_cell_size = hidden_dims[-1]
        decoder_input_size = input.get_shape()[2].value

        if encoder_mask is None:
            encoder_mask = tf.ones([batch_size, attention_len], tf.float32)

        encoder_states = tf.expand_dims(encoder_states, 2)  # [batch_size, attention_len, 1, attention_size]

        # v * tanh(attn_h + attn_s + b_attn), att_h = W_h*h_i, att_s = W_s*s_i
        w_h = tf.get_variable('W_h', [1, 1, attention_size, attention_size])
        b_attn = tf.get_variable('b_attn', [attention_size])
        v = tf.get_variable('v', [attention_size])
        att_H = tf.nn.conv2d(encoder_states, w_h, [1, 1, 1, 1], 'SAME')

        w_out = tf.get_variable('w_out', [attention_size+decoder_cell_size, decoder_cell_size])
        b_out = tf.get_variable('b_out', [decoder_cell_size],)

        decoder_inputs = tf.split(input, n_decoder_cells, axis=1)
        attention_distributions = tf.TensorArray(tf.float32, size=n_decoder_cells)
        outputs = tf.TensorArray(tf.float32, size=n_decoder_cells)

        state = initial_state
        for i in range(n_decoder_cells):
            inp = tf.reshape(decoder_inputs[i], [batch_size, decoder_input_size])

            output, state = cell(inp, state)
            reuse = None
            if i > 0:
                reuse = True
            with tf.variable_scope('attention', reuse=reuse):
                w_s = tf.get_variable('Ws', [decoder_cell_size, attention_size])
                att_s = tf.matmul(output, w_s)
                e = tf.reduce_sum(v * tf.tanh(att_H+att_s+b_attn), [2, 3])
                attention_distrib = tf.nn.softmax(e) * encoder_mask
                totals = tf.reduce_sum(attention_distrib, 1)
                attention_distrib = attention_distrib/tf.reshape(totals, [-1, 1])  # [batch_size, attention_len]
                attention_distributions = attention_distributions.write(i, attention_distrib)
                context_vec = tf.reshape(attention_distrib, [batch_size, attention_len, 1, 1]) * encoder_states
                context_vec = tf.reduce_sum(context_vec, [1, 2])  # [batch_size, attention_size]
                output = tf.matmul(tf.concat([context_vec, output], 1), w_out) + b_out
                outputs = outputs.write(i, output)
        tf.transpose(attention_distributions.stack(), [1, 0, 2], name='attention_distributions')
        outputs = tf.transpose(outputs.stack(), [1, 0, 2], name='outputs')

        tf.identity(state, name='last_state')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(outputs[:, -1, ], name='output')
        return output
