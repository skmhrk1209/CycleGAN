from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import argparse
import functools
import itertools
import time
import cv2
import cycle_gan
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="monet2photo_cycle_gan_model", help="model directory")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--buffer_size", type=int, default=1000, help="buffer size to shuffle dataset")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
parser.add_argument('--data_format', type=str, default="channels_last", help="data_format")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


class ImagePool(object):

    def __init__(self, max_size):

        self.max_size = max_size
        self.images = []

    def __call__(self, image):

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


filenames_A = tf.placeholder(dtype=tf.string, shape=[None])
filenames_B = tf.placeholder(dtype=tf.string, shape=[None])
batch_size = tf.placeholder(dtype=tf.int64, shape=[])
num_epochs = tf.placeholder(dtype=tf.int64, shape=[])
buffer_size = tf.placeholder(dtype=tf.int64, shape=[])

generator = cycle_gan.Model.Generator()
discriminator = cycle_gan.Model.Discriminator()

training = tf.placeholder(dtype=tf.bool, shape=[])
cycle_coefficient = tf.constant(value=10.0, dtype=tf.float32)

reals_A_iterator = dataset.input(
    filenames=filenames_A,
    batch_size=batch_size,
    num_epochs=num_epochs,
    buffer_size=buffer_size
)

reals_A = reals_A_iterator.get_next()

fakes_B_A = generator(
    inputs=reals_A,
    filters=32,
    residual_blocks=9,
    data_format=args.data_format,
    training=training,
    name="generator_B",
    reuse=False
)

fake_histories_B_A = tf.placeholder(
    dtype=fakes_B_A.dtype,
    shape=fakes_B_A.shape
)

fakes_A_A = generator(
    inputs=fakes_B_A,
    filters=32,
    residual_blocks=9,
    data_format=args.data_format,
    training=training,
    name="generator_A",
    reuse=False
)
real_logits_A = discriminator(
    inputs=reals_A,
    filters=64,
    layers=3,
    data_format=args.data_format,
    training=training,
    name="discriminator_A",
    reuse=False
)

fake_logits_B = discriminator(
    inputs=fakes_B_A,
    filters=64,
    layers=3,
    data_format=args.data_format,
    training=training,
    name="discriminator_B",
    reuse=False
)

fake_history_logits_B = discriminator(
    inputs=fake_histories_B_A,
    filters=64,
    layers=3,
    data_format=args.data_format,
    training=training,
    name="discriminator_B",
    reuse=True
)

reals_B_iterator = dataset.input(
    filenames=filenames_B,
    batch_size=batch_size,
    num_epochs=num_epochs,
    buffer_size=buffer_size
)

reals_B = reals_B_iterator.get_next()

fakes_A_B = generator(
    inputs=reals_B,
    filters=32,
    residual_blocks=9,
    data_format=args.data_format,
    training=training,
    name="generator_A",
    reuse=True
)

fake_histories_A_B = tf.placeholder(
    dtype=fakes_A_B.dtype,
    shape=fakes_A_B.shape
)

fakes_B_B = generator(
    inputs=fakes_A_B,
    filters=32,
    residual_blocks=9,
    data_format=args.data_format,
    training=training,
    name="generator_B",
    reuse=True
)

real_logits_B = discriminator(
    inputs=reals_B,
    filters=64,
    layers=3,
    data_format=args.data_format,
    training=training,
    name="discriminator_B",
    reuse=True
)

fake_logits_A = discriminator(
    inputs=fakes_A_B,
    filters=64,
    layers=3,
    data_format=args.data_format,
    training=training,
    name="discriminator_A",
    reuse=True
)

fake_history_logits_A = discriminator(
    inputs=fake_histories_A_B,
    filters=64,
    layers=3,
    data_format=args.data_format,
    training=training,
    name="discriminator_A",
    reuse=True
)

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

generator_global_step = tf.Variable(initial_value=0, trainable=False)
discriminator_global_step = tf.Variable(initial_value=0, trainable=False)

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

image_pool_A = ImagePool(50)
image_pool_B = ImagePool(50)

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

            start = time.time()

            session.run(
                [reals_A_iterator.initializer, reals_B_iterator.initializer],
                feed_dict={
                    filenames_A: ["data/monet2photo/monet/train.tfrecord"],
                    filenames_B: ["data/monet2photo/photo/train.tfrecord"],
                    batch_size: args.batch_size,
                    num_epochs: args.num_epochs,
                    buffer_size: args.buffer_size
                }
            )

            for i in itertools.count():

                fakes_B_A_, fakes_A_B_, _ = session.run(
                    [fakes_B_A, fakes_A_B, generator_train_op],
                    feed_dict={
                        training: True
                    }
                )

                fake_histories_B_A_ = image_pool_B(fakes_B_A_)
                fake_histories_A_B_ = image_pool_A(fakes_A_B_)

                print(fakes_B_A_.shape)
                print(fake_histories_B_A_.shape)

                session.run(
                    [discriminator_train_op],
                    feed_dict={
                        fake_histories_B_A: fake_histories_B_A_,
                        fake_histories_A_B: fake_histories_A_B_,
                        training: True
                    }
                )

                if i % 100 == 0:

                    generator_global_step_, generator_loss_ = session.run(
                        [generator_global_step, generator_loss],
                        feed_dict={
                            training: True
                        }
                    )

                    print("global_step: {}, generator_loss: {}".format(
                        generator_global_step_,
                        generator_loss_
                    ))

                    discriminator_global_step_, discriminator_loss_ = session.run(
                        [discriminator_global_step, discriminator_loss],
                        feed_dict={
                            fake_histories_B_A: fake_histories_B_A_,
                            fake_histories_A_B: fake_histories_A_B_,
                            training: True
                        }
                    )

                    print("global_step: {}, discriminator_loss: {}".format(
                        discriminator_global_step_,
                        discriminator_loss_
                    ))

                    checkpoint = saver.save(
                        sess=session,
                        save_path=os.path.join(args.model_dir, "model.ckpt"),
                        global_step=generator_global_step_
                    )

                    stop = time.time()

                    print("{} saved ({} sec)".format(
                        checkpoint,
                        stop - start
                    ))

                    start = time.time()

        except tf.errors.OutOfRangeError:

            print("training ended")

    if args.predict:

        try:

            print("prediction started")

            session.run(
                [reals_A_iterator.initializer, reals_B_iterator.initializer],
                feed_dict={
                    filenames_A: ["data/monet2photo/monet/test.tfrecord"],
                    filenames_B: ["data/monet2photo/photo/test.tfrecord"],
                    batch_size: args.batch_size,
                    num_epochs: args.num_epochs,
                    buffer_size: args.buffer_size
                }
            )

            for i in itertools.count():

                reals_A_, fakes_B_A_, reals_B_, fakes_A_B_ = session.run(
                    [reals_A, fakes_B_A, reals_B, fakes_A_B],
                    feed_dict={
                        training: False
                    }
                )

                images = np.concatenate([
                    np.concatenate([reals_A_, fakes_B_A_], axis=2),
                    np.concatenate([reals_B_, fakes_A_B_], axis=2),
                ], axis=1)

                images = utils.scale(images, -1, 1, 0, 1)

                for image in images:

                    cv2.imshow("image", image)

                    if cv2.waitKey(1000) == ord("q"):
                        break

        except tf.errors.OutOfRangeError:

            print("prediction ended")
