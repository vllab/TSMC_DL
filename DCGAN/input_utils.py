import tensorflow as tf
import os
import fnmatch
import numpy as np


def get_image_paths(im_dir, fname_pattern):
    '''
    Recusively get all file paths ending with `image_suffix` under `dir`
    dir: directory path
    return: list of image paths
    '''
    paths = []
    for root, dirnames, filenames in os.walk(im_dir):
        for filename in fnmatch.filter(filenames, fname_pattern):
            paths.append(os.path.join(root, filename))
    return paths


def get_input_batch_tensor(input_reader_fn,
                           inputpaths,
                           batch_size,
                           shuffle=True,
                           num_epochs=None,
                           path_queue_capacity=200,
                           data_queue_capacity=None,
                           min_after_dequeue=30,
                           num_threads=1):
    with tf.variable_scope('batch_input_producer'):
        path_queue = tf.train.string_input_producer(inputpaths,
                                                    capacity=path_queue_capacity,
                                                    shuffle=shuffle,
                                                    num_epochs=num_epochs)
        if not data_queue_capacity:
            data_queue_capacity = min_after_dequeue + 3 * batch_size
        return tf.train.batch(input_reader_fn(path_queue),
                              batch_size,
                              num_threads=num_threads,
                              capacity=data_queue_capacity,
                              enqueue_many=True)