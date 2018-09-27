from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import cycle_gan
import resnet
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="monet2photo_cycle_gan_model", help="model directory")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--buffer_size", type=int, default=1000, help="buffer size to shuffle dataset")
parser.add_argument('--data_format', type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


class Dataset(dataset.Dataset):

    def __init__(self, data_format):

        self.data_format = data_format

        super(Dataset, self).__init__()

    def parse(self, example):

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

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = utils.scale(image, 0, 1, -1, 1)

        if self.data_format == "channels_first":

            image = tf.transpose(image, [2, 0, 1])

        return image


cycle_gan_model = cycle_gan.Model(
    dataset_A=Dataset(args.data_format),
    dataset_B=Dataset(args.data_format),
    generator=resnet.Generator(
        filters=32,
        residual_blocks=9,
        data_format=args.data_format
    ),
    discriminator=resnet.Discriminator(
        filters=64,
        layers=3,
        data_format=args.data_format
    ),
    hyper_param=cycle_gan.Model.HyperParam(
        cycle_coefficient=10.0,
        identity_coefficient=5.0
    )
)

if args.train:

    cycle_gan_model.train(
        model_dir=args.model_dir,
        filenames_A=["data/monet2photo/monet/train.tfrecord"],
        filenames_B=["data/monet2photo/photo/train.tfrecord"],
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        buffer_size=args.buffer_size,
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            ),
            log_device_placement=False,
            allow_soft_placement=True
        )
    )

if args.predict:

    cycle_gan_model.predict(
        model_dir=args.model_dir,
        filenames_A=["data/monet2photo/monet/test.tfrecord"],
        filenames_B=["data/monet2photo/photo/test.tfrecord"],
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        buffer_size=args.buffer_size,
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            ),
            log_device_placement=False,
            allow_soft_placement=True
        )
    )
