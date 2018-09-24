from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import layers


class Model(object):

    """ implementation of Cycle GAN in TensorFlow

    [1] [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks]
        (https://arxiv.org/pdf/1703.10593.pdf) by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros, Mar 2017.
    """

    class Generator(object):

        def __call__(self, inputs, filters, data_format, training, name="generator", reuse=False):

            with tf.variable_scope(name, reuse=reuse):

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=filters << 0,
                    kernel_size=7,
                    strides=1,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.relu,
                    data_format=data_format,
                    training=training
                )

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=filters << 1,
                    kernel_size=3,
                    strides=2,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.relu,
                    data_format=data_format,
                    training=training
                )

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=filters << 2,
                    kernel_size=3,
                    strides=2,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.relu,
                    data_format=data_format,
                    training=training
                )

                for i in range(0, 9):

                    inputs = layers.residual_block(
                        inputs=inputs,
                        filters=filters << 2,
                        strides=1,
                        normalization=layers.instance_normalization,
                        activation=tf.nn.relu,
                        data_format=data_format,
                        training=training
                    )

                inputs = layers.deconvolutional_block(
                    inputs=inputs,
                    filters=filters << 1,
                    kernel_size=3,
                    strides=2,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.relu,
                    data_format=data_format,
                    training=training
                )

                inputs = layers.deconvolutional_block(
                    inputs=inputs,
                    filters=filters << 0,
                    kernel_size=3,
                    strides=2,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.relu,
                    data_format=data_format,
                    training=training
                )

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=3,
                    kernel_size=7,
                    strides=1,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.tanh,
                    data_format=data_format,
                    training=training
                )

                return inputs

    class Discriminator(object):

        def __call__(self, filters, inputs, data_format, training, name="discriminator", reuse=False):

            with tf.variable_scope(name, reuse=reuse):

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=filters << 0,
                    kernel_size=4,
                    strides=2,
                    normalization=None,
                    activation=tf.nn.leaky_relu,
                    data_format=data_format,
                    training=training
                )

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=filters << 1,
                    kernel_size=4,
                    strides=2,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.leaky_relu,
                    data_format=data_format,
                    training=training
                )

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=filters << 2,
                    kernel_size=4,
                    strides=2,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.leaky_relu,
                    data_format=data_format,
                    training=training
                )

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=filters << 3,
                    kernel_size=4,
                    strides=1,
                    normalization=layers.instance_normalization,
                    activation=tf.nn.leaky_relu,
                    data_format=data_format,
                    training=training
                )

                inputs = layers.convolutional_block(
                    inputs=inputs,
                    filters=1,
                    kernel_size=4,
                    strides=1,
                    normalization=None,
                    activation=None,
                    data_format=data_format,
                    training=training
                )

                return inputs
