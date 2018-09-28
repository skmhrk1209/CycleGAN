from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import collections
import os
import itertools
import time
import cv2


class Model(object):

    """ implementation of Cycle GAN in TensorFlow

    [1] [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks]
        (https://arxiv.org/pdf/1703.10593.pdf) by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros, Mar 2017.
    """

    HyperParam = collections.namedtuple(
        "HyperParam", (
            "cycle_coefficient",
            "identity_coefficient",
            "learning_rate",
            "beta1",
            "beta2"
        )
    )

    def __init__(self, dataset_A, dataset_B, generator, discriminator, hyper_param):

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.generator = generator
        self.discriminator = discriminator
        self.hyper_param = hyper_param

        self.training = tf.placeholder(dtype=tf.bool, shape=[])

        self.next_reals_A = self.dataset_A.get_next()
        self.next_reals_B = self.dataset_B.get_next()

        self.reals_A = tf.placeholder(dtype=tf.float32, shape=self.next_reals.shape)
        self.reals_B = tf.placeholder(dtype=tf.float32, shape=self.next_reals.shape)

        self.fakes_B_A = self.generator(
            inputs=self.reals_A,
            training=self.training,
            name="generator_B",
            reuse=False
        )

        self.fakes_A_B_A = self.generator(
            inputs=self.fakes_B_A,
            training=self.training,
            name="generator_A",
            reuse=False
        )

        self.fakes_A_A = self.generator(
            inputs=self.reals_A,
            training=self.training,
            name="generator_A",
            reuse=True
        )

        self.real_logits_A = self.discriminator(
            inputs=self.reals_A,
            training=self.training,
            name="discriminator_A",
            reuse=False
        )

        self.fake_logits_B = self.discriminator(
            inputs=self.fakes_B_A,
            training=self.training,
            name="discriminator_B",
            reuse=False
        )

        self.reals_B = self.dataset_B.input()

        self.fakes_A_B = self.generator(
            inputs=self.reals_B,
            training=self.training,
            name="generator_A",
            reuse=True
        )

        self.fakes_B_A_B = self.generator(
            inputs=self.fakes_A_B,
            training=self.training,
            name="generator_B",
            reuse=True
        )

        self.fakes_B_B = self.generator(
            inputs=self.reals_B,
            training=self.training,
            name="generator_B",
            reuse=True
        )

        self.real_logits_B = self.discriminator(
            inputs=self.reals_B,
            training=self.training,
            name="discriminator_B",
            reuse=True
        )

        self.fake_logits_A = self.discriminator(
            inputs=self.fakes_A_B,
            training=self.training,
            name="discriminator_A",
            reuse=True
        )

        self.generator_loss = \
            tf.reduce_mean(tf.square(self.fake_logits_A - tf.ones_like(self.fake_logits_A))) + \
            tf.reduce_mean(tf.square(self.fake_logits_B - tf.ones_like(self.fake_logits_B))) + \
            tf.reduce_mean(tf.abs(self.reals_A - self.fakes_A_B_A)) * self.cycle_coefficient + \
            tf.reduce_mean(tf.abs(self.reals_B - self.fakes_B_A_B)) * self.cycle_coefficient + \
            tf.reduce_mean(tf.abs(self.reals_A - self.fakes_A_A)) * self.identity_coefficient + \
            tf.reduce_mean(tf.abs(self.reals_B - self.fakes_B_B)) * self.identity_coefficient \

        self.discriminator_loss = \
            tf.reduce_mean(tf.square(self.real_logits_A - tf.ones_like(self.real_logits_A))) + \
            tf.reduce_mean(tf.square(self.real_logits_B - tf.ones_like(self.real_logits_B))) + \
            tf.reduce_mean(tf.square(self.fake_logits_A - tf.zeros_like(self.fake_logits_A))) + \
            tf.reduce_mean(tf.square(self.fake_logits_B - tf.zeros_like(self.fake_logits_B))) \

        self.generator_variables = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_A") + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_B") \

        self.discriminator_variables = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_A") + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_B") \

        self.generator_global_step = tf.Variable(initial_value=0, trainable=False)
        self.discriminator_global_step = tf.Variable(initial_value=0, trainable=False)

        self.generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hyper_param.learning_rate, beta1=self.hyper_param.beta1, beta2=self.hyper_param.beta2
        )
        self.discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hyper_param.learning_rate, beta1=self.hyper_param.beta1, beta2=self.hyper_param.beta2
        )

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(self.update_ops):

            self.generator_train_op = self.generator_optimizer.minimize(
                loss=self.generator_loss,
                var_list=self.generator_variables,
                global_step=self.generator_global_step
            )

            self.discriminator_train_op = self.discriminator_optimizer.minimize(
                loss=self.discriminator_loss,
                var_list=self.discriminator_variables,
                global_step=self.discriminator_global_step
            )

    def initialize(self, model_dir):

        session = tf.get_default_session()

        session.run(tf.local_variables_initializer())
        print("local variables initialized")

        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(model_dir)

        if checkpoint:
            saver.restore(session, checkpoint)
            print(checkpoint, "loaded")

        else:
            session.run(tf.global_variables_initializer())
            print("global variables initialized")

        return saver

    def train(self, model_dir, filenames_A, filenames_B, batch_size, num_epochs, buffer_size, config):

        with tf.Session(config=config) as session:

            saver = self.initialize(model_dir)

            try:

                print("training started")

                start = time.time()

                self.dataset_A.initialize(
                    filenames=filenames_A,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    buffer_size=buffer_size
                )

                self.dataset_B.initialize(
                    filenames=filenames_B,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    buffer_size=buffer_size
                )

                for i in itertools.count():

                    reals_A, reals_B = session.run(
                        [self.next_reals_A, self.next_reals_B]
                    )

                    feed_dict = {
                        self.reals_A: reals_A,
                        self.reals_B: reals_B,
                        self.training: True
                    }

                    session.run(self.generator_train_op, feed_dict=feed_dict)
                    session.run(self.discriminator_train_op, feed_dict=feed_dict)

                    if i % 100 == 0:

                        generator_global_step, generator_loss = session.run(
                            [self.generator_global_step, self.generator_loss],
                            feed_dict=feed_dict
                        )

                        print("global_step: {}, generator_loss: {:.2f}".format(
                            generator_global_step,
                            generator_loss
                        ))

                        discriminator_global_step, discriminator_loss = session.run(
                            [self.discriminator_global_step, self.discriminator_loss],
                            feed_dict=feed_dict
                        )

                        print("global_step: {}, discriminator_loss: {:.2f}".format(
                            discriminator_global_step,
                            discriminator_loss
                        ))

                        checkpoint = saver.save(
                            sess=session,
                            save_path=os.path.join(model_dir, "model.ckpt"),
                            global_step=generator_global_step
                        )

                        stop = time.time()

                        print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))

                        start = time.time()

                        fakes_B_A, fakes_A_B = session.run(
                            [self.fakes_B_A, self.fakes_A_B],
                            feed_dict=feed_dict
                        )

                        images = np.concatenate([
                            np.concatenate([reals_A, fakes_B_A], axis=2),
                            np.concatenate([reals_B, fakes_A_B], axis=2),
                        ], axis=1)

                        for image in images:

                            cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                            cv2.waitKey(100)

            except tf.errors.OutOfRangeError:

                print("training ended")

    def predict(self, model_dir, filenames_A, filenames_B, batch_size, num_epochs, buffer_size, config):

        with tf.Session(config=config) as session:

            self.initialize(model_dir)

            try:

                print("prediction started")

                self.dataset_A.initialize(
                    filenames=filenames_A,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    buffer_size=buffer_size
                )

                self.dataset_B.initialize(
                    filenames=filenames_B,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    buffer_size=buffer_size
                )

                for i in itertools.count():

                    reals_A, reals_B = session.run(
                        [self.next_reals_A, self.next_reals_B]
                    )

                    feed_dict = {
                        self.reals_A: reals_A,
                        self.reals_B: reals_B,
                        self.training: True
                    }

                    fakes_B_A, fakes_A_B = session.run(
                        [self.fakes_B_A, self.fakes_A_B],
                        feed_dict=feed_dict
                    )

                    images = np.concatenate([
                        np.concatenate([reals_A, fakes_B_A], axis=2),
                        np.concatenate([reals_B, fakes_A_B], axis=2),
                    ], axis=1)

                    for image in images:

                        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                        cv2.waitKey(100)

            except tf.errors.OutOfRangeError:

                print("prediction ended")
