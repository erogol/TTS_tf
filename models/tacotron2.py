import io
import os
import re
import time
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from tensorflow import keras

from TTS_tf.layers.tacotron2 import Encoder, Decoder, Postnet
from TTS_tf.utils.tf_utils import shape_list


class Tacotron2(keras.models.Model):
    def __init__(self, 
                 num_chars,
                 num_speakers,
                 r=5,
                 mel_dim=80,
                 memory_size=5,
                 attn_win=False,
                 gst=False,
                 attn_norm="sigmoid",
                 prenet_type="original",
                 prenet_dropout=True,
                 use_forward_attn=False,
                 use_trans_agent=False,
                 use_forward_attn_mask=False,
                 use_loc_attn=True,
                 separate_stopnet=True):
        super(Tacotron2, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.gst = gst
        self.num_speakers = num_speakers
        self.speaker_embed_dim = 256

        self.embedding = keras.layers.Embedding(num_chars, 512)
        self.encoder = Encoder(512)
        # TODO: most of the decoder args have no use at the momment
        self.decoder = Decoder(mel_dim, r, 'location', False, 'sigmoid', 'normal', True, False, False, False, True, 2, False, 0)
        self.postnet = Postnet(mel_dim, 5)

    def call(self, characters, text_lengths, frames):
        B, T = shape_list(characters)
        embedding_vectors = self.embedding(characters)
        encoder_output = self.encoder(embedding_vectors)
        decoder_states = self.decoder.build_decoder_initial_states(B, 512, T)
        decoder_frames, stop_tokens, attentions = self.decoder.decode(encoder_output, frames, decoder_states, text_lengths)
        decoder_frames = tf.reshape(decoder_frames, [B, -1, self.mel_dim])
        postnet_frames = self.postnet(decoder_frames)
        output_frames = decoder_frames + postnet_frames
        return decoder_frames, output_frames, attentions, stop_tokens



