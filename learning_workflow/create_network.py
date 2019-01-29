import sys
sys.path.append('..')
import genetor
import glob
import tensorflow as tf

session = tf.Session()


noise = tf.placeholder(
    shape = [None, 100],
    dtype = tf.float32,
    name = 'noise')

architecture = [{
    'type': 'fc',
    'input': noise,
    'params': {
        'units': 7 * 7 * 64
    }
}, {
    'type': 'reshape',
    'params': {
        'shape': [-1, 7, 7, 64]
    }
}, *genetor.builder.new_architecture(
    model = 'deconv',
    structure = {
        'type': 'upsampling',
        'filters': [64, 1],
        'strides': [2, 2]
    }
), {
    'type': 'sigmoid',
    'output_label': 'output'
}]

with tf.variable_scope('generator'):
    generator_output = genetor.builder.new_graph(architecture = architecture)


input = genetor.builder.new_graph(architecture = [{
    'type': 'tf_data',
    'params': {
        'meta_path': '../data/tf_records/mnist/meta.json',
        'parsers': {
            'input': genetor.components.parse_image(shape = [28, 28, 1])
        },
        'create_placeholders_for': ['input'],
        'return': 'input'
    }
}])
architecture = [{
    'type': 'concat',
    'input': input,
    'params': {
        'other': [generator_output],
        'axis': 0
    }
}, *genetor.builder.new_architecture(
    model = 'cnn',
    structure = {
        'filters': [32, 64],
        'kernels': [5, 5],
        'units': [256, 1]
    }
), {
    'type': 'sigmoid',
    'name': 'output'
}]

with tf.variable_scope('discriminator'):
    discriminator_output = genetor.builder.new_graph(architecture = architecture)

genetor.builder.new_graph(architecture = [{
    'type': 'gan_loss',
    'input': discriminator_output
}])

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')



