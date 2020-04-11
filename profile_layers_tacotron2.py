import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

from TTS_tf.layers.tacotron2 import ConvBNBlock, Postnet, Encoder
from TTS_tf.layers.common_layers import LocationSensitiveAttention


np.random.seed(1)
tf.random.set_seed(1)
tf.executing_eagerly()
tf.config.list_physical_devices('GPU')
use_cuda = tf.test.is_gpu_available()
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


B = 32
T_en = 100
T_de = 500
D_em = 512
D_en = 512


def create_rnd_input(shape, dtype=np.float32):
    dummy_input = tf.convert_to_tensor(np.random.rand(*shape).astype(np.float32))
    return dummy_input


def run_layer(layer, x, num_runs=1):
    def _run_layer(layer, x):"?P"
        if type(x) == list:
            o = layer(**x)
        else:
            o = layer(x)
        return 0
    layer_name = type(layer).__name__
    time_start = time.time()
    o = _run_layer(layer, x)
    time_first_run = time.time() - time_start
    time_start = time.time()
    for i in range(num_runs):
        o = _run_layer(layer, x)
    time_total = time.time() - time_start
    time_avg = time_total / num_runs
    print(f'{layer_name} -- time_avg: {time_avg} -- first_run:{time_first_run}')
    o_normal = o

    layer = tf.function(layer)
    time_start = time.time()
    o = _run_layer(layer, x)
    time_first_run = time.time() - time_start
    time_start = time.time()
    for i in range(num_runs):
        o = _run_layer(layer, x)
    time_total = time.time() - time_start
    time_avg = time_total / num_runs
    print(f'{layer_name}.function -- time_avg: {time_avg} -- first_run:{time_first_run}')
    assert tf.reduce_sum(o_normal - o ) == 0

x = create_rnd_input([B, T_en, D_em])
layer = ConvBNBlock(512, 5, 'relu')
out = run_layer(layer, x, 50)

layer = Postnet(512, 5)
out = run_layer(layer, x, 5)

layer = Encoder(512)
out = run_layer(layer, x, 5)

memory = create_rnd_input([B, T_en, D_en])
alignment_old = create_rnd_input([B, T_en, 1])
alignment_cum = create_rnd_input([B, T_en, 1])
states = tf.TensorArray()
layer = LocationSensitiveAttention(units=128,
                                   memory=memory,
                                   memory_sequence_length=None,
                                   normalize=False,
                                   probability_fn="softmax",
                                   kernel_initializer="glorot_uniform",
                                   dtype=None,
                                   location_attention_filters=32,
                                   location_attention_kernel_size=31)
out = run_layer(layer, [memory, [aligntment_cum, alignment_old]])
