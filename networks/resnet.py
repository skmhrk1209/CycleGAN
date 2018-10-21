import tensorflow as tf
from . import ops


class Generator(object):

    def __init__(self, filters, residual_blocks, data_format):

        self.filters = filters
        self.residual_blocks = residual_blocks
        self.data_format = data_format

    def __call__(self, inputs, training, name="generator", reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.conv2d(
                inputs=inputs,
                filters=self.filters << 0,
                kernel_size=[7, 7],
                strides=[1, 1],
                data_format=self.data_format,
                name="conv2d_{}".format(0)
            )

            inputs = ops.instance_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="instance_normalization_{}".format(0)
            )

            inputs = tf.nn.relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=self.filters << 1,
                kernel_size=[3, 3],
                strides=[2, 2],
                data_format=self.data_format,
                name="conv2d_{}".format(1)
            )

            inputs = ops.instance_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="instance_normalization_{}".format(1)
            )

            inputs = tf.nn.relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=self.filters << 2,
                kernel_size=[3, 3],
                strides=[2, 2],
                data_format=self.data_format,
                name="conv2d_{}".format(2)
            )

            for i in range(3, self.residual_blocks + 3):

                inputs = ops.residual_block(
                    inputs=inputs,
                    filters=self.filters << 2,
                    strides=[1, 1],
                    normalization=ops.instance_normalization,
                    activation=tf.nn.relu,
                    data_format=self.data_format,
                    training=training,
                    name="residual_block_{}".format(i)
                )

            inputs = ops.instance_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="instance_normalization_{}".format(i)
            )

            inputs = tf.nn.relu(inputs)

            inputs = ops.deconv2d(
                inputs=inputs,
                filters=self.filters << 1,
                kernel_size=[3, 3],
                strides=[2, 2],
                data_format=self.data_format,
                name="deconv2d_{}".format(i + 1)
            )

            inputs = ops.instance_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="instance_normalization_{}".format(i + 1)
            )

            inputs = tf.nn.relu(inputs)

            inputs = ops.deconv2d(
                inputs=inputs,
                filters=self.filters << 0,
                kernel_size=[3, 3],
                strides=[2, 2],
                data_format=self.data_format,
                name="deconv2d_{}".format(i + 2)
            )

            inputs = ops.instance_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="instance_normalization_{}".format(i + 2)
            )

            inputs = tf.nn.relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=3,
                kernel_size=[7, 7],
                strides=[1, 1],
                data_format=self.data_format,
                name="conv2d_{}".format(i + 3)
            )

            inputs = ops.instance_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="instance_normalization_{}".format(i + 3)
            )

            inputs = tf.nn.sigmoid(inputs)

            return inputs


class Discriminator(object):

    def __init__(self, filters, layers, data_format):

        self.filters = filters
        self.layers = layers
        self.data_format = data_format

    def __call__(self, inputs, training, name="discriminator", reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.conv2d(
                inputs=inputs,
                filters=self.filters,
                kernel_size=[4, 4],
                strides=[2, 2],
                data_format=self.data_format,
                name="conv2d_{}".format(0)
            )

            inputs = tf.nn.leaky_relu(inputs)

            for i in range(1, self.layers):

                inputs = ops.conv2d(
                    inputs=inputs,
                    filters=self.filters << i,
                    kernel_size=[4, 4],
                    strides=[2, 2],
                    data_format=self.data_format,
                    name="conv2d_{}".format(i)
                )

                inputs = ops.instance_normalization(
                    inputs=inputs,
                    data_format=self.data_format,
                    training=training,
                    name="instance_normalization_{}".format(i)
                )

                inputs = tf.nn.leaky_relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=self.filters << (i + 1),
                kernel_size=[4, 4],
                strides=[1, 1],
                data_format=self.data_format,
                name="conv2d_{}".format(i + 1)
            )

            inputs = ops.instance_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="instance_normalization_{}".format(i + 1)
            )

            inputs = tf.nn.leaky_relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=1,
                kernel_size=[4, 4],
                strides=[1, 1],
                data_format=self.data_format,
                name="conv2d_{}".format(i + 2)
            )

            return inputs
