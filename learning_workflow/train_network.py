import sys
sys.path.append('..')
import glob
import os
import tensorflow as tf
import genetor
import numpy as np


def noise_generator(iteration_n, batch_size):
    return np.random.randn(batch_size, 100)

def is_real_generator(iteration_n, batch_size):
    return [1.] * batch_size + [0.] * batch_size


trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    record_paths = glob.glob(f'../data/tf_records/mnist/*.tfrecords'),
    placeholders = {
        'batch_size:0': 100,
        'noise:0': noise_generator,
        'gan_loss_0/is_real:0': is_real_generator
    },
    optimizers = ['generator_optimizer', 'discriminator_optimizer'],
    summary = {
        'path': '../trained_models/summaries/mnist',
        'scalars': ['gan_loss_0/generator_loss:0', 'gan_loss_0/discriminator_loss:0'],
        'images': [{
            'name': 'generated',
            'tensor': 'generator/output:0',
            'max_outputs': 4
        }]
    }
)

trainer.train_epoch()






