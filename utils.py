from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def scale(input, input_min, input_max, output_min, output_max):

    return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)
