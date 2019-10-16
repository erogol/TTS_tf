import os
import copy
import torch
import unittest
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(3)

from TTS_tf.utils.generic_utils import load_config, count_parameters
from TTS_tf.layers.losses import loss_l1, stopnet_loss
from TTS_tf.models.tacotron import Tacotron

#pylint: disable=unused-variable

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))


class TacotronTrainTest(unittest.TestCase):
    @staticmethod
    def test_train_step():
        input_dummy = np.random.rand(8, 128).astype('float32')
        input_lengths = np.random.randint(100, 129, (8, ))
        input_lengths[-1] = 128
        mel_spec = np.random.rand(8, 30, c.audio['num_mels']).astype('float32')
        linear_spec = np.random.rand(8, 30, c.audio['num_freq']).astype('float32')
        mel_lengths = np.random.randint(20, 30, (8,))
        mel_lengths[-1] = 30
        stop_targets = np.zeros([8, 30, 1]).astype('float32')
        speaker_ids = np.random.randint(0, 5, (8, ))

        for idx in mel_lengths:
            stop_targets[:, int(idx.item()):, 0] = 1.0

        stop_targets = stop_targets.reshape(input_dummy.shape[0],
                                         stop_targets.shape[1] // c.r, -1)
        stop_targets = tf.squeeze(stop_targets.sum(2) > 0.0)

        criterion = loss_l1
        criterion_st = stopnet_loss
        model = Tacotron(
            num_chars=32,
            num_speakers=5,
            linear_dim=c.audio['num_freq'],
            mel_dim=c.audio['num_mels'],
            r=c.r,
            memory_size=c.memory_size
        )  #FIXME: missing num_speakers parameter to Tacotron ctor
        # dummy call
        _ = model(input_dummy, input_lengths, mel_spec)
        print(" > Num parameters for Tacotron model:%s" %
              (count_parameters(model, c)))
        model_ref = copy.deepcopy(model.trainable_variables)
        count = 0
        # check if model and model_ref parameters are equal
        for param, param_ref in zip(model.trainable_variables,
                                    model_ref):
            assert tf.reduce_sum(param - param_ref) == 0, param
            count += 1
        # train model couple of iterations
        optimizer = tf.keras.optimizers.Adam()
        for _ in range(5):
            with tf.GradientTape() as tape:
                mel_out, linear_out, align, stop_tokens = model(
                    input_dummy, input_lengths, mel_spec)
                loss = criterion(mel_out, mel_spec, mel_lengths)
                stop_loss = criterion_st(stop_tokens, tf.cast(stop_targets,tf.float32))
                loss = loss + criterion(linear_out, linear_spec,
                                        mel_lengths) + stop_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.trainable_variables,
                                    model_ref):
            # ignore pre-higway layer since it works conditional
            # if count not in [145, 59]:
            assert (param.numpy() != param_ref.numpy()).any(
            ), "param {} with shape {} not updated!! \n{}\n{}".format(
                count, param.shape, param, param_ref)
            count += 1


