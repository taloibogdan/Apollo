from __future__ import division
import glob
import os
import tensorflow as tf


CKPT_META_PATH = '../trained_models/checkpoints/mnist_base/ckpt.meta'
session = tf.Session()
saver = tf.train.import_meta_graph(CKPT_META_PATH)
saver.restore(session, tf.train.latest_checkpoint(
    os.path.dirname(CKPT_META_PATH)
))
graph = tf.get_default_graph()

input = graph.get_tensor_by_name('input:0')
target = graph.get_tensor_by_name('target:0')
optimizer = graph.get_operation_by_name('optimizer')
iterator_initializer = graph.get_operation_by_name('iterator_initializer')
record_paths_tensor = graph.get_tensor_by_name('record_paths:0')
next_batch = graph.get_operation_by_name('next_batch')
batch_size = graph.get_tensor_by_name('batch_size:0')
accuracy_mean = graph.get_tensor_by_name('cross_entropy_0/accuracy_mean:0')

TF_RECORDS_PATHS = [
    glob.glob(f'../data/tf_records/mnist/{usage}/*')
    for usage in ['train', 'val', 'test']
]
TRAIN_PATHS, VAL_PATHS, TEST_PATHS = TF_RECORDS_PATHS
session.run(
    iterator_initializer,
    feed_dict = {
        record_paths_tensor: TRAIN_PATHS,
        batch_size: 1000
    }
)

for _ in range(10000):
    _, _, acc = session.run(
        [next_batch, optimizer, accuracy_mean]
    )
    print(acc)





