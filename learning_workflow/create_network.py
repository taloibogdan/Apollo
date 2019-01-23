from __future__ import division
import sys
sys.path.append('..')

import genetor
import glob
import tensorflow as tf

architecture = []
session = tf.Session()

def parse_example(example_proto):
    feature_description = {
        'input': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64)
    }
    
    example = tf.parse_single_example(example_proto, feature_description)

    example['input'] = tf.image.decode_png(example['input'], channels = 1)
    example['input'] = tf.image.resize_images(example['input'], [28, 28]) / 255.

    return example

record_paths_tensor = tf.placeholder(
    dtype = tf.string,
    shape = [None],
    name = 'record_paths'
)

dataset = tf.data.TFRecordDataset(record_paths_tensor)
dataset = dataset.map(parse_example)
dataset = dataset.repeat()
batch_size = tf.placeholder(
    dtype = tf.int64,
    shape = [],
    name = 'batch_size'
)
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
iterator_initializer = iterator.make_initializer(
    dataset,
    name = 'iterator_initializer'
)

next_batch = iterator.get_next(name = 'next_batch')

input = tf.placeholder_with_default(
    input = next_batch['input'],
    shape = [None, 28, 28, 1],
    name = 'input'
)
target = tf.placeholder_with_default(
    input = next_batch['target'],
    shape = [None],
    name = 'target'
)

filters = [32, 64]
kernels = [5, 5]
for f, k in zip(filters, kernels):
    architecture += [{
        'type': 'conv',
        'params': {
            'filters': f,
            'kernel_size': k
        }
    }, {
        'type': 'max_pool'
    }]
architecture += [{
    'type': 'flatten'
}]

units = [256, 10]
for u in units:
    architecture += [{
        'type': 'fc',
        'params': {
            'units': u
        }
    }]
architecture += [{
    'type': 'cross_entropy',
    'params': {
        'target': target
    }
}]

loss = genetor.builder.new_graph(architecture = architecture, input = input)
optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')



saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist_base/ckpt')



