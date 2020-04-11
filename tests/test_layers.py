import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from TTS_tf.layers.tacotron import BatchNormConv1d, Highway, CBHG, Encoder, Decoder
from TTS_tf.layers.common_layers import Prenet, Attention
from TTS_tf.layers.losses import loss_l1

#pylint: disable=unused-variable

#class BatchNormConv1dTests(unittest.TestCase):
#    def test_in_out(self):
#        layer = BatchNormConv1d(128, 5, 1, 'same', keras.layers.ReLU())
#        dummy_input = np.random.rand(4, 10, 32).astype(np.float32)
#        output = layer(dummy_input)
#        assert output.shape[0] == 4
#        assert output.shape[1] == 10
#        assert output.shape[2] == 128
#
#        layer = BatchNormConv1d(128, 5, 1, 'same', None)
#        dummy_input = np.random.rand(4, 10, 32).astype(np.float32)
#        output = layer(dummy_input)
#        assert output.shape[0] == 4
#        assert output.shape[1] == 10
#        assert output.shape[2] == 128
#
#
#class PrenetTests(unittest.TestCase):
#    def test_in_out(self):
#        layer = Prenet(128, units=[256, 128])
#        dummy_input = np.random.rand(4, 128).astype(np.float32)
#
#        output = layer(dummy_input)
#        assert output.shape[0] == 4
#        assert output.shape[1] == 128
#
#
#class AttentionTests(unittest.TestCase):
#    def test_in_out(self):
#        # vanilla attention
#        layer = Attention(attn_dim=128,
#                          use_loc_attn=False,
#                          loc_attn_n_filters=0,
#                          loc_attn_kernel_size=0,
#                          use_windowing=False,
#                          norm='sigmoid',
#                          use_forward_attn=False,
#                          use_trans_agent=False,
#                          use_forward_attn_mask=False)
#        values = np.random.rand(4, 16, 128).astype(np.float32)
#        query = np.random.rand(4, 256).astype(np.float32)
#        layer.process_values(values)
#        context_vector = layer(query, values)
#        attention_weights = layer.attn_weights
#
#        eps = 1e-5
#        assert np.abs(np.sum(attention_weights) - 4) < eps, np.sum(attention_weights)
#        assert np.abs(np.sum(attention_weights[0]) - 1) < eps, np.sum(attention_weights[0])
#        assert attention_weights.shape[0] == 4
#        assert attention_weights.shape[1] == 16
#        assert context_vector.ndim ==  2
#        assert context_vector.shape[0] == 4
#        assert context_vector.shape[1] == 128
#
#        # location attention
#        layer = Attention(attn_dim=128,
#                          use_loc_attn=True,
#                          loc_attn_n_filters=32,
#                          loc_attn_kernel_size=31,
#                          use_windowing=False,
#                          norm='sigmoid',
#                          use_forward_attn=False,
#                          use_trans_agent=False,
#                          use_forward_attn_mask=False)
#        values = np.random.rand(4, 16, 128).astype(np.float32)
#        query = np.random.rand(4, 256).astype(np.float32)
#        layer.init_states(values)
#        layer.process_values(values)
#        context_vector = layer(query, values)
#        attention_weights = layer.attn_weights
#
#        eps = 1e-5
#        assert np.abs(np.sum(attention_weights) - 4) < eps, np.sum(attention_weights)
#        assert np.abs(np.sum(attention_weights[0]) - 1) < eps, np.sum(attention_weights[0])
#        assert attention_weights.shape[0] == 4
#        assert attention_weights.shape[1] == 16
#        assert context_vector.ndim ==  2
#        assert context_vector.shape[0] == 4
#        assert context_vector.shape[1] == 128
#        assert np.sum(layer.attn_weights) > 0 
#        assert np.sum(layer.attn_weights_cum) > 0
#
#        # masking
#        layer = Attention(attn_dim=128,
#                          use_loc_attn=True,
#                          loc_attn_n_filters=32,
#                          loc_attn_kernel_size=31,
#                          use_windowing=False,
#                          norm='sigmoid',
#                          use_forward_attn=False,
#                          use_trans_agent=False,
#                          use_forward_attn_mask=False)
#        values = np.random.rand(4, 16, 128).astype(np.float32)
#        query = np.random.rand(4, 256).astype(np.float32)
#        value_lens = np.random.randint(8, 16, size=4)
#        value_lens[-1] = 16
#        mask = tf.sequence_mask(value_lens)
#        layer.init_states(values)
#        layer.process_values(values)
#        context_vector = layer(query, values, mask)
#        attention_weights = layer.attn_weights
#
#        eps = 1e-5
#        assert np.abs(np.sum(attention_weights) - 4) < eps, np.sum(attention_weights)
#        assert np.abs(np.sum(attention_weights[0]) - 1) < eps, np.sum(attention_weights[0])
#        assert attention_weights.shape[0] == 4
#        assert attention_weights.shape[1] == 16
#        assert context_vector.ndim ==  2
#        assert context_vector.shape[0] == 4
#        assert context_vector.shape[1] == 128
#        assert np.sum(layer.attn_weights) > 0 
#        assert np.sum(layer.attn_weights_cum) > 0
#
#        # TODO: forward attention
#
#        # TODO: windowing
#
#
#class HighwayTests(unittest.TestCase):
#    def test_in_out(self):
#        layer = Highway(128)
#        dummy_input = np.random.rand(4, 128).astype(np.float32)
#
#        output = layer(dummy_input)
#        assert output.shape[0] == 4
#        assert output.shape[1] == 128
#
#
#class CBHGTests(unittest.TestCase):
#    def test_in_out(self):
#        #pylint: disable=attribute-defined-outside-init
#        layer = CBHG(
#            K=8,
#            conv_bank_filters=80,
#            conv_filters=[160, 128],
#            highway_units=80,
#            gru_units=80,
#            num_highways=4)
#        dummy_input = np.random.rand(4, 8, 128).astype(np.float32)
#
#        print(layer)
#        output = layer(dummy_input)
#        assert output.shape[0] == 4
#        assert output.shape[1] == 8
#        assert output.shape[2] == 160
#
#
#class EncoderTests(unittest.TestCase):
#    def test_in_out(self):
#        #pylint: disable=attribute-defined-outside-init
#        layer = Encoder()
#        dummy_input = np.random.rand(4, 8, 128).astype(np.float32)
#
#        output = layer(dummy_input)
#        assert output.shape[0] == 4
#        assert output.shape[1] == 8
#        assert output.shape[2] == 256


class DecoderTests(unittest.TestCase):
    @staticmethod
    def test_in_out():
        breakpoint()
        layer = Decoder(
            input_dim=256,
            memory_dim=80,
            r=2,
            memory_size=0,
            use_attn_windowing=False,
            attn_norm="sigmoid",
            prenet_type='original',
            prenet_dropout=True,
            use_forward_attn=False,
            use_trans_agent=False,
            use_forward_attn_mask=False,
            use_loc_attn=True,
            separate_stopnet=True,
            speaker_embedding_dim=0)
        dummy_input = tf.convert_to_tensor(np.random.rand(4, 128, 256).astype(np.float32))
        dummy_memory = tf.convert_to_tensor(np.random.rand(4, 450, 80).astype(np.float32))

        output, alignment, stop_tokens = layer(dummy_input, dummy_memory, mask=None)

#        assert output.shape[0] == 4
#        assert output.shape[1] == 10, "size not {}".format(output.shape[1])
#        assert output.shape[2] == 80, "size not {}".format(output.shape[2])
#        assert stop_tokens.shape[0] == 4
#
    # @staticmethod
    # def test_in_out_multispeaker():
    #     layer = Decoder(
    #         in_features=256,
    #         memory_dim=80,
    #         r=2,
    #         memory_size=4,
    #         attn_windowing=False,
    #         attn_norm="sigmoid",
    #         prenet_type='original',
    #         prenet_dropout=True,
    #         forward_attn=True,
    #         use_trans_agent=True,
    #         use_forward_attn_mask=True,
    #         location_attn=True,
    #         separate_stopnet=True,
    #         speaker_embedding_dim=80)
    #     dummy_input = np.random.rand(4, 8, 256)
    #     dummy_memory = np.random.rand(4, 2, 80)
    #     dummy_embed = np.random.rand(4, 80)

    #     output, alignment, stop_tokens = layer(
    #         dummy_input, dummy_memory, mask=None, speaker_embeddings=dummy_embed)

    #     assert output.shape[0] == 4
    #     assert output.shape[1] == 1, "size not {}".format(output.shape[1])
    #     assert output.shape[2] == 80 * 2, "size not {}".format(output.shape[2])
    #     assert stop_tokens.shape[0] == 4


# class EncoderTests(unittest.TestCase):
#     def test_in_out(self):
#         layer = Encoder(128)
#         dummy_input = np.random.rand(4, 8, 128)

#         print(layer)
#         output = layer(dummy_input)
#         print(output.shape)
#         assert output.shape[0] == 4
#         assert output.shape[1] == 8
#         assert output.shape[2] == 256  # 128 * 2 BiRNN


#class L1LossTests(unittest.TestCase):
#    def test_in_out(self):
#        dummy_input = np.ones([4, 8, 128], dtype=np.float32)
#        dummy_target = np.ones([4, 8, 128], dtype=np.float32)
#        dummy_length = (np.ones(4) * 8)
#        output = loss_l1(dummy_input, dummy_target, dummy_length)
#        assert output == 0.0
#
#        dummy_input = np.ones([4, 8, 128], dtype=np.float32)
#        dummy_target = np.zeros([4, 8, 128], dtype=np.float32)
#        dummy_length = (np.ones(4) * 8)
#        output = loss_l1(dummy_input, dummy_target, dummy_length)
#        assert output == 1.0, "1.0 vs {}".format(output.data[0])
