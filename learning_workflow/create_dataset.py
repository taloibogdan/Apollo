import sys
sys.path.append('..')
from genetor.dataset.tf_records import TFWriter
import os
import tensorflow as tf
import glob


for usage in ['train', 'val', 'test']:
    writer = TFWriter(
        dir = f'../data/tf_records/mnist/{usage}/',
        format = {
            'input': 'bytes',
            'target': 'int'
        },
    )

    for filepath in glob.glob(f'../data/raw/mnist_{usage}/*'):
        writer.write_sample({
            'input': writer.read_im(filepath),
            'target': int(os.path.basename(filepath).split('_')[0])
        })

