import tensorflow as tf
import numpy as np
import collections
import os
import itertools
import time
import cv2


class Model(object):

    def __init__(self, dataset_A, dataset_B, generator, discriminator, hyper_params, name="gan", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            self.name = name
            self.dataset_A = dataset_A
            self.dataset_B = dataset_B
            self.generator = generator
            self.discriminator = discriminator
            self.hyper_param = hyper_params

            self.training = tf.placeholder(dtype=tf.bool, shape=[])

            self.next_reals_A = self.dataset_A.get_next()
            self.next_reals_B = self.dataset_B.get_next()

            self.reals_A = tf.placeholder(dtype=tf.float32, shape=self.next_reals_A.shape)
            self.reals_B = tf.placeholder(dtype=tf.float32, shape=self.next_reals_B.shape)

            self.fakes_B_A = self.generator(
                inputs=self.reals_A,
                training=self.training,
                name="generator_B"
            )

            self.fakes_A_B_A = self.generator(
                inputs=self.fakes_B_A,
                training=self.training,
                name="generator_A"
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
                name="discriminator_A"
            )

            self.fake_logits_B = self.discriminator(
                inputs=self.fakes_B_A,
                training=self.training,
                name="discriminator_B"
            )

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
                tf.reduce_mean(tf.abs(self.reals_A - self.fakes_A_B_A)) * self.hyper_param.cycle_coefficient + \
                tf.reduce_mean(tf.abs(self.reals_B - self.fakes_B_A_B)) * self.hyper_param.cycle_coefficient + \
                tf.reduce_mean(tf.abs(self.reals_A - self.fakes_A_A)) * self.hyper_param.identity_coefficient + \
                tf.reduce_mean(tf.abs(self.reals_B - self.fakes_B_B)) * self.hyper_param.identity_coefficient \

            self.discriminator_loss = \
                tf.reduce_mean(tf.square(self.real_logits_A - tf.ones_like(self.real_logits_A))) + \
                tf.reduce_mean(tf.square(self.real_logits_B - tf.ones_like(self.real_logits_B))) + \
                tf.reduce_mean(tf.square(self.fake_logits_A - tf.zeros_like(self.fake_logits_A))) + \
                tf.reduce_mean(tf.square(self.fake_logits_B - tf.zeros_like(self.fake_logits_B))) \

            self.generator_variables = \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{}/generator_A".format(self.name)) + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{}/generator_B".format(self.name)) \

            self.discriminator_variables = \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{}/discriminator_A".format(self.name)) + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="{}/discriminator_B".format(self.name)) \

            self.generator_global_step = tf.Variable(initial_value=0, trainable=False)
            self.discriminator_global_step = tf.Variable(initial_value=0, trainable=False)

            self.generator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_param.learning_rate,
                beta1=self.hyper_param.beta1,
                beta2=self.hyper_param.beta2
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_param.learning_rate,
                beta1=self.hyper_param.beta1,
                beta2=self.hyper_param.beta2
            )

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

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

            self.saver = tf.train.Saver()

            self.summary = tf.summary.merge([
                tf.summary.image("reals_A", self.reals_A),
                tf.summary.image("fakes_B_A", self.fakes_B_A),
                tf.summary.image("fakes_A_B_A", self.fakes_A_B_A),
                tf.summary.image("fakes_A_A", self.fakes_A_A),
                tf.summary.image("reals_B", self.reals_B),
                tf.summary.image("fakes_A_B", self.fakes_A_B),
                tf.summary.image("fakes_B_A_B", self.fakes_B_A_B),
                tf.summary.image("fakes_B_B", self.fakes_B_B),
                tf.summary.scalar("generator_loss", self.generator_loss),
                tf.summary.scalar("discriminator_loss", self.discriminator_loss)
            ])

    def initialize(self):

        session = tf.get_default_session()

        checkpoint = tf.train.latest_checkpoint(self.name)

        if checkpoint:
            self.saver.restore(session, checkpoint)
            print(checkpoint, "loaded")

        else:
            global_variables = tf.global_variables(scope=self.name)
            session.run(tf.variables_initializer(global_variables))
            print("global variables in {} initialized".format(self.name))

    def train(self, filenames_A, filenames_B, num_epochs, batch_size, buffer_size):

        session = tf.get_default_session()
        writer = tf.summary.FileWriter(self.name, session.graph)

        print("training started")

        start = time.time()

        self.dataset_A.initialize(
            filenames=filenames_A,
            num_epochs=num_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size
        )

        self.dataset_B.initialize(
            filenames=filenames_B,
            num_epochs=num_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size
        )

        for i in itertools.count():

            try:
                reals_A, reals_B = session.run(
                    [self.next_reals_A, self.next_reals_B]
                )

            except tf.errors.OutOfRangeError:
                print("training ended")
                break

            feed_dict = {
                self.reals_A: reals_A,
                self.reals_B: reals_B,
                self.training: True
            }

            session.run(
                [self.generator_train_op, self.discriminator_train_op],
                feed_dict=feed_dict
            )

            generator_global_step, discriminator_global_step = session.run(
                [self.generator_global_step, self.discriminator_global_step]
            )

            if generator_global_step % 100 == 0:

                generator_loss, discriminator_loss = session.run(
                    [self.generator_loss, self.discriminator_loss],
                    feed_dict=feed_dict
                )

                print("global_step: {}, generator_loss: {:.2f}".format(
                    generator_global_step,
                    generator_loss
                ))
                print("global_step: {}, discriminator_loss: {:.2f}".format(
                    discriminator_global_step,
                    discriminator_loss
                ))

                summary = session.run(self.summary, feed_dict=feed_dict)
                writer.add_summary(summary, global_step=generator_global_step)

                if generator_global_step % 100000 == 0:

                    checkpoint = self.saver.save(
                        sess=session,
                        save_path=os.path.join(self.name, "model.ckpt"),
                        global_step=generator_global_step
                    )

                    tf.train.write_graph(
                        graph_or_graph_def=session.graph.as_graph_def(),
                        logdir=self.name,
                        name="graph.pb",
                        as_text=False
                    )

                    stop = time.time()
                    print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))
                    start = time.time()
