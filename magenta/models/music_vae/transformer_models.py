# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LSTM-based encoders and decoders for MusicVAE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from magenta.common import flatten_maybe_padded_sequences
from magenta.common import Nade
from magenta.models.music_vae import base_model
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest


class TransformerEncoder(base_model.BaseEncoder):

  def output_depth(self):
    pass

  def build(self, hparams, is_training=True):
    pass

  def encode(self, sequence, sequence_length):
    pass


class TransformerDecoder(base_model.BaseDecoder):

  def build(self, hparams, output_depth, is_training=True):
    pass

  def reconstruction_loss(self, x_input, x_target, x_length, z=None, c_input=None):
    pass

  def sample(self, n, max_length=None, z=None, c_input=None):
    pass