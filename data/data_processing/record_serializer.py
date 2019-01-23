import tensorflow as tf


class RecordSerializer(object):

    def __init__(self, params):
        self.params = params

    @staticmethod
    def _create_bytes_feature(value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

    @staticmethod
    def _create_float_feature(value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

    @staticmethod
    def _create_int64_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


    def serialize_example(*args, **kwargs):
        pass

    def write_records(*args, **kwargs):
        pass




    



