import math
import os
import shutil
import tensorflow as tf

class Coordinator(object):

    def __init__(self, ckpt_meta_path, record_paths, placeholders, summary,
                 optimizers = ['optimizer'],
                 record_paths_placeholder = 'record_paths:0',
                 batch_size = 'batch_size:0',
                 iterator_initializer = 'iterator_initializer',
                 next_batch = 'next_batch'):

        self.ckpt_meta_path = ckpt_meta_path
        self.record_paths = record_paths
        self.record_paths_placeholder = record_paths_placeholder
        self.batch_size = batch_size
        self.placeholders = placeholders
        self.summary = summary
        self.optimizers = optimizers
        self.iterator_initializer = iterator_initializer
        self.next_batch = next_batch

        self.load_session()
        self.load_tensors()
        self.load_operations()
        self.create_summary()

        self.session.run(tf.global_variables_initializer())

        self.epoch_n = -1


    def train(self, n_iterations = 1, validate_every = 1, log_every = 'iteration'):
        self.epoch_n = 0

        self.initialize_iterators('train') 
        n_iterations = math.ceil(self.n_samples / self.placeholders[self.batch_size])
        for iteration_n in range(n_iterations):
            results = self.session.run(
                [self.summary_merged] + 
                [self.operations[self.next_batch]] + 
                [self.operations[optimizer_name] for optimizer_name in self.optimizers]
            )
            self.summary_writer.add_summary(
                results[0],
                (self.epoch_n * n_iterations) + iteration_n
            )


    def create_summary(self):
        tf.summary.scalar('accuracy', self.tensors[self.summary['scalars'][0]])

        if os.path.exists(self.summary['path']):
            shutil.rmtree(self.summary['path'])

        self.summary_merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(
            self.summary['path'],
            self.graph
        )



    def load_session(self):
        self.session = tf.Session()
        self.saver = tf.train.import_meta_graph(self.ckpt_meta_path)
        self.saver.restore(self.session, tf.train.latest_checkpoint(
            os.path.dirname(self.ckpt_meta_path)
        ))
        self.graph = tf.get_default_graph()


    def load_tensors(self):
        self.tensors = dict()
        tensor_names = [
            *self.placeholders.keys(),
            self.record_paths_placeholder,
            *self.summary['scalars'],
            *self.summary['images']
        ]
        for tensor_name in tensor_names:
            self.tensors[tensor_name] = self.graph.get_tensor_by_name(
                tensor_name
            )


    def load_operations(self):
        self.operations = dict()
        operation_names = [
            *self.optimizers,
            self.next_batch,
            self.iterator_initializer
        ]
        for operation_name in operation_names:
            self.operations[operation_name] = self.graph.get_operation_by_name(
                operation_name
            )


    def initialize_iterators(self, usage):
        self.n_samples = sum(
            sum(1 for _ in tf.python_io.tf_record_iterator(filename))
            for filename in self.record_paths[usage]
        )
        self.session.run(
            self.operations[self.iterator_initializer],
            feed_dict = {
                self.tensors[self.record_paths_placeholder]: self.record_paths[usage],
                self.tensors[self.batch_size]: self.placeholders[self.batch_size]
            }
        )
        
    
