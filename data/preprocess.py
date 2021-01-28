import tensorflow as tf
import numpy as np


def create_dataset(generator_fn, output_types, output_shapes, batch_size, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_generator(
        generator_fn, output_types, output_shapes
    )

    boundaries = [256 * (i+1) for i in range(20)]
    batch_sizes = [batch_size] * (len(boundaries) + 1)

    dataset = dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(
            _element_length_fn,
            boundaries,
            batch_sizes,
            drop_remainder=True,
            pad_to_bucket_boundary=False
        )
    ).shuffle(shuffle_buffer)
    return dataset


def load_POP909_dataset_completor(dataset_path, key, batch_size):
    data = np.load(dataset_path, allow_pickle=True)
    melodies = [d[key] for d in data]

    def generator_fn():
        for melody in melodies:
            yield tf.convert_to_tensor([melody])

    dataset = create_dataset(generator_fn, tf.int64, tf.TensorShape([1, None]), batch_size)
    return dataset


def load_POP909_dataset_translator(dataset_path, key1, key2, batch_size):
    data = np.load(dataset_path, allow_pickle=True)
    key1_data = [d[key1] for d in data]
    key2_data = [d[key2] for d in data]
    
    def generator_fn():
        for key1_d, key2_d  in zip(key1_data, key2_data):
            yield tf.convert_to_tensor([key1_d]), tf.convert_to_tensor([key2_d])
    
    output_shapes = (tf.TensorShape([1, None]), tf.TensorShape([1, None]))
    output_types = (tf.int64, tf.int64)
    dataset = create_dataset(generator_fn, output_types, output_shapes, batch_size)
    return dataset


def _element_length_fn(x, y=None):
    return tf.python.ops.array_ops.shape(x)[1]
