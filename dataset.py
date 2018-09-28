from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Dataset(object):

    def __init__(self):

        self.filenames = tf.placeholder(dtype=tf.string, shape=[None])
        self.batch_size = tf.placeholder(dtype=tf.int64, shape=[])
        self.num_epochs = tf.placeholder(dtype=tf.int64, shape=[])
        self.buffer_size = tf.placeholder(dtype=tf.int64, shape=[])

        self.dataset = tf.data.TFRecordDataset(self.filenames)
        self.dataset = self.dataset.shuffle(self.buffer_size)
        self.dataset = self.dataset.repeat(self.num_epochs)
        self.dataset = self.dataset.map(self.parse)
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_initializable_iterator()

    def parse(self, example):

        raise NotImplementedError()

    def initialize(self, filenames, batch_size, num_epochs, buffer_size):

        session = tf.get_default_session()

        session.run(
            [self.iterator.initializer],
            feed_dict={
                self.filenames: filenames,
                self.batch_size: batch_size,
                self.num_epochs: num_epochs,
                self.buffer_size: buffer_size
            }
        )

    def get_next(self):

        return self.iterator.get_next()
