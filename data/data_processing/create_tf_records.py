from record_serializer import RecordSerializer
import os
import tensorflow as tf

class Serializer(RecordSerializer):

    @staticmethod
    def extract_label(image_path):
        label = int(os.path.basename(image_path).split('_')[0])
        return label

    def serialize_example(self, image_path):
        im = open(image_path, 'rb').read()
        label = Serializer.extract_label(image_path)

        feature = {
          'input': RecordSerializer._create_bytes_feature(im),
          'target': RecordSerializer._create_int64_feature(label),
        }

        example_proto = tf.train.Example(
          features = tf.train.Features(feature = feature)
        )
        return example_proto.SerializeToString()

    def write_records(self):
        input_paths = self.params['input_paths']
        output_paths = self.params['output_paths']
        for input_path, output_path in zip(input_paths, output_paths):

            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            writer = tf.python_io.TFRecordWriter(output_path)
            for filename in os.listdir(input_path):
                example = self.serialize_example(
                    os.path.join(input_path, filename)
                )
                writer.write(example)


if __name__ == '__main__':
    folders = ['train', 'val', 'test']
    rs = Serializer({
        'input_paths': [f'../raw/mnist_{usage}' for usage in folders],
        'output_paths': [f'../tf_records/mnist/{usage}/record.tfrecords' for usage in folders]
    })
    rs.write_records()

