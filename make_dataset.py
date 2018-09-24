from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import glob
import os
import sys
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str)
parser.add_argument("filename", type=str)
args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.filename) as writer:

    for label, directory in enumerate(sorted(glob.glob(os.path.join(args.directory, "*")))):

        for file in glob.glob(os.path.join(directory, "*")):

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
