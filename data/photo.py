from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import dataset


class Dataset(dataset.Dataset):

    def __init__(self, data_format):

        self.data_format = data_format

        super(Dataset, self).__init__()

    def parse(self, example):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string,
                    default_value=""
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        if self.data_format == "channels_first":

            image = tf.transpose(image, [2, 0, 1])

        return image
