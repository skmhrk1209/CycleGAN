#=================================================================================================#
# Implementation of Cycle GAN
#
# 2018/10/01 Hiroki Sakuma
# (https://github.com/skmhrk1209/GAN)
#
# original papers
# [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks]
# (https://arxiv.org/pdf/1703.10593.pdf)
#=================================================================================================#

import tensorflow as tf
import argparse
from models import cycle_gan
from networks import resnet
from data import monet, photo
from utils import attr_dict

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="monet2photo_cycle_gan_model", help="model directory")
parser.add_argument('--filenames_A', type=str, nargs="+", default=["monet_train.tfrecord"], help="tfrecord filenames for domain A")
parser.add_argument('--filenames_B', type=str, nargs="+", default=["photo_train.tfrecord"], help="tfrecord filenames for domain B")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--buffer_size", type=int, default=1000, help="buffer size to shuffle dataset")
parser.add_argument('--data_format', type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


cycle_gan_model = cycle_gan.Model(
    dataset_A=monet.Dataset(args.data_format),
    dataset_B=photo.Dataset(args.data_format),
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
    hyper_params=attr_dict.AttrDict(
        cycle_coefficient=10.0,
        identity_coefficient=5.0,
        learning_rate=0.0002,
        beta1=0.9,
        beta2=0.999
    ),
    name=args.model_dir
)

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.gpu,
        allow_growth=True
    ),
    log_device_placement=False,
    allow_soft_placement=True
)

with tf.Session() as session:

    cycle_gan_model.initialize()

    if args.train:

        cycle_gan_model.train(
            filenames_A=args.filenames_A,
            filenames_B=args.filenames_B,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size
        )
