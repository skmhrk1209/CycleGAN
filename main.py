from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys
import argparse
import functools
import itertools
import cv2
import cycle_gan
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="monet2photo_cycle_gan_model", help="model directory")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
parser.add_argument('--data_format', type=str, default="channels_last", help="data_format")
parser.add_argument('--pool_size', type=int, default=50, help="image pool size")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


class ImagePool(object):

    def __init__(self, max_size):

        self.max_size = max_size
        self.images = []

    def __call__(self, image):

        if self.max_size <= 0:
            return image

        if len(self.images) < self.max_size:
            self.images.append(image)
            return image

        if np.random.rand() > 0.5:
            index = np.random.randint(0, self.max_size)
            history = self.images[index]
            self.images[index] = image
            return history
        else:
            return image


def parse_fn(example):

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
    image = utils.scale(image, 0., 255., -1., 1.)

    return image


filenames_A = tf.placeholder(dtype=tf.string, shape=[None])
filenames_B = tf.placeholder(dtype=tf.string, shape=[None])
batch_size = tf.placeholder(dtype=tf.int64, shape=[])
num_epochs = tf.placeholder(dtype=tf.int64, shape=[])
buffer_size = tf.placeholder(dtype=tf.int64, shape=[])
data_format = tf.placeholder(dtype=tf.string, shape=[])
training = tf.placeholder(dtype=tf.bool, shape=[])
cycle_coefficient = tf.constant(value=10.0, dtype=tf.float32)

dataset_A = tf.data.TFRecordDataset(filenames_A)
dataset_A = dataset_A.shuffle(buffer_size)
dataset_A = dataset_A.repeat(num_epochs)
dataset_A = dataset_A.map(parse_fn)
dataset_A = dataset_A.map(preprocess)
dataset_A = dataset_A.batch(batch_size)
dataset_A = dataset_A.prefetch(1)
iterator_A = dataset_A.make_initializable_iterator()

dataset_B = tf.data.TFRecordDataset(filenames_B)
dataset_B = dataset_B.shuffle(buffer_size)
dataset_B = dataset_B.repeat(num_epochs)
dataset_B = dataset_B.map(parse_fn)
dataset_B = dataset_B.map(preprocess)
dataset_B = dataset_B.batch(batch_size)
dataset_B = dataset_B.prefetch(1)
iterator_B = dataset_B.make_initializable_iterator()

generator = cycle_gan.Model.Generator()
discriminator = cycle_gan.Model.Discriminator()

reals_A = iterator_A.get_next()
fakes_B_A = generator(inputs=reals_A, filters=32, data_format=args.data_format,
                      training=training, name="generator_B", reuse=False)
fake_histories_B_A = tf.placeholder(dtype=fakes_B_A.dtype, shape=fakes_B_A.shape)
fakes_A_A = generator(inputs=fakes_B_A, filters=32, data_format=args.data_format,
                      training=training, name="generator_A", reuse=False)
real_logits_A = discriminator(inputs=reals_A, filters=64, data_format=args.data_format,
                              training=training, name="discriminator_A", reuse=False)
fake_logits_B = discriminator(inputs=fakes_B_A, filters=64, data_format=args.data_format,
                              training=training, name="discriminator_B", reuse=False)
fake_history_logits_B = discriminator(inputs=fake_histories_B_A, filters=64, data_format=args.data_format,
                                      training=training, name="discriminator_B", reuse=True)

reals_B = iterator_B.get_next()
fakes_A_B = generator(inputs=reals_B, filters=32, data_format=args.data_format,
                      training=training, name="generator_A", reuse=True)
fake_histories_A_B = tf.placeholder(dtype=fakes_A_B.dtype, shape=fakes_A_B.shape)
fakes_B_B = generator(inputs=fakes_A_B, filters=32, data_format=args.data_format,
                      training=training, name="generator_B", reuse=True)
real_logits_B = discriminator(inputs=reals_B, filters=64, data_format=args.data_format,
                              training=training, name="discriminator_B", reuse=True)
fake_logits_A = discriminator(inputs=fakes_A_B, filters=64, data_format=args.data_format,
                              training=training, name="discriminator_A", reuse=True)
fake_history_logits_A = discriminator(inputs=fake_histories_A_B, filters=64, data_format=args.data_format,
                                      training=training, name="discriminator_A", reuse=True)

generator_loss = \
    tf.reduce_mean(tf.squared_difference(fake_logits_A, tf.ones_like(fake_logits_A))) + \
    tf.reduce_mean(tf.squared_difference(fake_logits_B, tf.ones_like(fake_logits_B))) + \
    tf.reduce_mean(tf.abs(reals_A - fakes_A_A)) * cycle_coefficient + \
    tf.reduce_mean(tf.abs(reals_B - fakes_B_B)) * cycle_coefficient \

discriminator_loss = \
    tf.reduce_mean(tf.squared_difference(real_logits_A, tf.ones_like(real_logits_A))) + \
    tf.reduce_mean(tf.squared_difference(real_logits_B, tf.ones_like(real_logits_B))) + \
    tf.reduce_mean(tf.squared_difference(fake_history_logits_A, tf.zeros_like(fake_history_logits_A))) + \
    tf.reduce_mean(tf.squared_difference(fake_history_logits_B, tf.zeros_like(fake_history_logits_B))) \

generator_variables = \
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_A") + \
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_B") \

discriminator_variables = \
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_A") + \
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_B") \

generator_global_step = tf.Variable(0, trainable=False)
discriminator_global_step = tf.Variable(0, trainable=False)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

    generator_train_op = tf.train.AdamOptimizer().minimize(
        loss=generator_loss,
        var_list=generator_variables,
        global_step=generator_global_step
    )

    discriminator_train_op = tf.train.AdamOptimizer().minimize(
        loss=discriminator_loss,
        var_list=discriminator_variables,
        global_step=discriminator_global_step
    )

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.gpu,
        allow_growth=True
    ),
    log_device_placement=False,
    allow_soft_placement=True
)

image_pool_A = ImagePool(args.pool_size)
image_pool_B = ImagePool(args.pool_size)

with tf.Session(config=config) as session:

    saver = tf.train.Saver()

    checkpoint = tf.train.latest_checkpoint(args.model_dir)

    session.run(tf.local_variables_initializer())

    print("local variables initialized")

    if checkpoint:

        saver.restore(session, checkpoint)

        print(checkpoint, "loaded")

    else:

        session.run(tf.global_variables_initializer())

        print("global variables initialized")

    if args.train:

        try:

            print("training started")

            feed_dict = {
                filenames_A: ["data/monet2photo/monet/train.tfrecord"],
                filenames_A: ["data/monet2photo/photo/train.tfrecord"],
                batch_size: args.batch_size,
                num_epochs: args.num_epochs,
                buffer_size: 10000
            }

            session.run([iterator_A.initializer, iterator_B.initializer], feed_dict=feed_dict)

            for i in itertools.count():

                feed_dict = {
                    training: True
                }

                fakes_B, fakes_A, _ = session.run([fakes_B_A, fakes_A_B, generator_train_op], feed_dict=feed_dict)

                feed_dict = {
                    fake_histories_B_A: image_pool_B(fakes_B),
                    fake_histories_A_B: image_pool_A(fakes_A),
                    training: True
                }

                session.run(discriminator_train_op, feed_dict=feed_dict)

                if i % 100 == 0:

                    checkpoint = saver.save(
                        session,
                        os.path.join(args.model_dir, "model.ckpt"),
                        global_step=generator_global_step
                    )

                    print(checkpoint, "saved")

        except tf.errors.OutOfRangeError:

            print("training ended")
