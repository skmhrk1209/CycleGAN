from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ops


class Generator(object):

    def __init__(self, filters, residual_blocks, data_format):

        self.filters = filters
        self.residual_blocks = residual_blocks
        self.data_format = data_format

    def __call__(self, inputs, training, name="generator", reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=self.filters << 0,
                kernel_size=7,
                strides=1,
                normalization=ops.instance_normalization,
                activation=tf.nn.relu,
                data_format=self.data_format,
                training=training
            )

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=self.filters << 1,
                kernel_size=3,
                strides=2,
                normalization=ops.instance_normalization,
                activation=tf.nn.relu,
                data_format=self.data_format,
                training=training
            )

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=self.filters << 2,
                kernel_size=3,
                strides=2,
                normalization=ops.instance_normalization,
                activation=tf.nn.relu,
                data_format=self.data_format,
                training=training
            )

            for _ in range(self.residual_blocks):

                inputs = ops.residual_block(
                    inputs=inputs,
                    filters=self.filters << 2,
                    strides=1,
                    normalization=ops.instance_normalization,
                    activation=tf.nn.relu,
                    data_format=self.data_format,
                    training=training
                )

            inputs = ops.deconv2d_block(
                inputs=inputs,
                filters=self.filters << 1,
                kernel_size=3,
                strides=2,
                normalization=ops.instance_normalization,
                activation=tf.nn.relu,
                data_format=self.data_format,
                training=training
            )

            inputs = ops.deconv2d_block(
                inputs=inputs,
                filters=self.filters << 0,
                kernel_size=3,
                strides=2,
                normalization=ops.instance_normalization,
                activation=tf.nn.relu,
                data_format=self.data_format,
                training=training
            )

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=3,
                kernel_size=7,
                strides=1,
                normalization=ops.instance_normalization,
                activation=tf.nn.tanh,
                data_format=self.data_format,
                training=training
            )

            return inputs


class Discriminator(object):

    def __init__(self, filters, layers, data_format):

        self.filters = filters
        self.layers = layers
        self.data_format = data_format

    def __call__(self, inputs, training, name="discriminator", reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=self.filters,
                kernel_size=4,
                strides=2,
                normalization=None,
                activation=tf.nn.leaky_relu,
                data_format=self.data_format,
                training=training
            )

            for i in range(1, self.layers):

                inputs = ops.conv2d_block(
                    inputs=inputs,
                    filters=self.filters << i,
                    kernel_size=4,
                    strides=2,
                    normalization=ops.instance_normalization,
                    activation=tf.nn.leaky_relu,
                    data_format=self.data_format,
                    training=training
                )

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=self.filters << self.layers,
                kernel_size=4,
                strides=1,
                normalization=ops.instance_normalization,
                activation=tf.nn.leaky_relu,
                data_format=self.data_format,
                training=training
            )

            inputs = ops.conv2d_block(
                inputs=inputs,
                filters=1,
                kernel_size=4,
                strides=1,
                normalization=None,
                activation=None,
                data_format=self.data_format,
                training=training
            )

            return inputs
