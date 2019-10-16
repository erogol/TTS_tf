import os
import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

from TTS_tf.utils.generic_utils import setup_model, load_config, check_gradient, count_parameters
from TTS_tf.utils.text.symbols import phonemes, symbols


file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))


def test_setup_model():
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)
    num_speakers = 0
    model = setup_model(num_chars, num_speakers, c)
    print(count_parameters(model, c))


def test_check_gradient():
    dummy_grad = np.random.rand(10, 100)
    grad_clip = 2.0
    grad, grad_norm = check_gradient(dummy_grad, grad_clip)
    assert np.abs(tf.norm(grad) - grad_clip) < 0.00001, tf.norm(grad)
