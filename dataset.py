from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import glob
import utils


def make(filename, directory):

    with tf.python_io.TFRecordWriter(filename) as writer:

        for label, sub_directory in enumerate(sorted(glob.glob(os.path.join(directory, "*")))):

            for file in glob.glob(os.path.join(sub_directory, "*")):

                writer.write(
                    record=tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[file.encode("utf-8")]
                                    )
                                ),
                                "label": tf.train.Feature(
                                    int64_list=tf.train.Int64List(
                                        value=[label]
                                    )
                                )
                            }
                        )
                    ).SerializeToString()
                )


def input(filenames, batch_size, num_epochs, buffer_size):

    def parse(example):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string,
                    default_value=""
                ),
                "label": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.int64,
                    default_value=0
                )
            }
        )

        return features["path"]

    def preprocess(path):

        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = utils.scale(image, 0., 1., -1., 1.)

        return image

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset.make_initializable_iterator()
