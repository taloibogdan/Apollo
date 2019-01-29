import sys
sys.path.append('..')
import glob
import os
import tensorflow as tf
from genetor.train.coordinator import Coordinator


TF_RECORDS_PATHS = {
    usage: glob.glob(f'../data/tf_records/mnist/{usage}/*.tfrecords')
    for usage in ['train', 'val', 'test']
}

trainer = Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist_base/ckpt.meta',
    record_paths = TF_RECORDS_PATHS,
    placeholders = {
        'batch_size:0': 10
    },
    summary = {
        'path': '../trained_models/summaries/mnist_base',
        'scalars': ['cross_entropy_0/accuracy_mean:0'],
        'images': []
    }
)

trainer.train()






