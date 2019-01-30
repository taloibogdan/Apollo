from . import components
import tensorflow as tf

DEFAULT_PARAMS = {
    'resize_up_conv': {
        'initialization': components.default_initialization,
        'filters': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 'SAME',
        'activation': tf.nn.relu
    },
    'up_conv': {
        'initialization': components.default_initialization,
        'filters': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 'SAME',
        'activation': tf.nn.relu
    },
    'conv': {
        'initialization': components.default_initialization,
        'filters': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 'SAME',
        'activation': tf.nn.relu
    },
    'dilated_conv': {
        'initialization': components.default_initialization,
        'filters': 32,
        'kernel_size': 3,
        'rate': 2,
        'padding': 'SAME',
        'activation': tf.nn.relu
    },
    'upsampling': {
        'filters': 32,
        'kernel_size': 3,
        'strides': ( 2, 2 ),
        'padding': 'same'
    },
    'concat': {
        'axis': -1
    },
    'max_pool': {
        'strides': [1, 2, 2, 1],
        'ksize': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'fc': {
        'initialization': components.default_initialization,
        'units': 512,
        'activation': tf.nn.relu
    },
    'dropout': {
        'keep_prob': 0.5
    },
    'caps': {
        'routing': {
            'method': 'agreement',
            'n_iterations': 2
        }
    },
    'caps_margin_loss': {
        'absent_threshold': 0.1,
        'absent_weight': 0.5,
        'present_threshold': 0.9
    },
    'argmax': {
        'axis': -1
    },
    'reduce_max': {
        'axis': -1
    },
    'gan_loss': {
        'generator_scope': 'generator',
        'discriminator_scope': 'discriminator'
    },
    'fc_parallel': {
        'initialization': components.default_initialization,
        'units': 512,
        'activation': tf.nn.relu
    },
    'scrambler': {
        'n_untouched': 0
    },
    'RNN': {
        'cell_type': 'LSTM',
        'hidden_dims': [64],
        'initial_state': None,
        'sequence_length': None
    },
    'bidirectional_RNN': {
        'cell_type': 'LSTM',
        'hidden_dims': [64],
        'initial_state_fw': None,
        'initial_state_bw': None,
        'sequence_length': None
    },
    'attention_decoder': {
        'cell_type': 'LSTM',
        'hidden_dims': [64],
        'initial_state': None,
        'encoder_states': None,
        'encoder_mask': None
    }
}

