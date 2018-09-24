from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def instance_normalization(inputs, data_format):

    return tf.contrib.layers.instance_norm(
        inputs=inputs,
        data_format="NCHW" if data_format == "channels_first" else "NHWC"
    )


def convolutional_block(inputs, filters, kernel_size, strides, normalization, activation, data_format, training):

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        data_format=data_format,
        kernel_initializer=tf.variance_scaling_initializer(),
    )

    if normalization:

        inputs = normalization(inputs, data_format)

    if activation:

        inputs = activation(inputs)

    return inputs


def deconvolutional_block(inputs, filters, kernel_size, strides, normalization, activation, data_format, training):

    inputs = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        data_format=data_format,
        kernel_initializer=tf.variance_scaling_initializer(),
    )

    if normalization:

        inputs = normalization(inputs, data_format)

    if activation:

        inputs = activation(inputs)

    return inputs


def residual_block(inputs, filters, strides, normalization, activation, data_format, training):

    shortcut = convolutional_block(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        normalization=normalization,
        activation=None,
        data_format=data_format,
        training=training
    )

    inputs = convolutional_block(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        normalization=normalization,
        activation=activation,
        data_format=data_format,
        training=training
    )

    inputs = convolutional_block(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        normalization=normalization,
        activation=None,
        data_format=data_format,
        training=training
    )

    inputs += shortcut

    inputs = activation(inputs)

    return inputs
