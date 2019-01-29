import sys
sys.path.append('..')
import genetor
import os
import tensorflow as tf
import glob


writer = genetor.data.Writer(
    dir = f'../data/tf_records/mnist/',
    format = {
        'input': 'bytes'
    },
)
for usage in ['train', 'val', 'test']:
    for filepath in glob.glob(f'../data/raw/mnist_{usage}/*'):
        writer.write_sample({
            'input': writer.read_im(filepath),
        })

