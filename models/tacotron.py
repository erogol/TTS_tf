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

from TTS_tf.layers.tacotron import Encoder, Decoder, Postnet

class Tacotron(tf.keras.Model):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r=5,
                 linear_dim=1025,
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
        super(Tacotron, self).__init__()
        # model arguments
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.gst = gst
        self.num_speakers = num_speakers
        self.speaker_embed_dim = 256
        # model layers
        self.embedding = keras.layers.Embedding(num_chars, 256)
        decoder_dim = 256 + self.speaker_embed_dim if num_speakers > 1 else 256
        encoder_dim = 256 + self.speaker_embed_dim if num_speakers > 1 else 256
        proj_speaker_dim = 80 if num_speakers > 1 else 0
        # boilerplate model
        self.encoder = Encoder()
        self.decoder = Decoder(decoder_dim, mel_dim, r, memory_size, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               use_forward_attn, use_trans_agent, use_forward_attn_mask,
                               use_loc_attn, separate_stopnet,
                               proj_speaker_dim)
        self.postnet = Postnet(mel_dim)
        self.last_linear = keras.layers.Dense(linear_dim)
        # speaker embedding layers
        if num_speakers > 1:
            self.speaker_embedding = keras.layers.Embedding(num_speakers, 256)
            self.speaker_project_mel = keras.models.Sequential(
                [keras.layers.Dense(proj_speaker_dim), keras.layers.Activation('tanh')])
            self.speaker_embeddings = None
            self.speaker_embeddings_projected = None
        # global style token layers
        if self.gst:
            gst_embedding_dim = 256
            # TODO: gst

    def _init_states(self):
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

    def compute_speaker_embedding(self, speaker_ids):
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(
                " [!] Model has speaker embedding layer but speaker_id is not provided"
            )
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            self.speaker_embeddings = self._compute_speaker_embedding(
                speaker_ids)
            self.speaker_embeddings_projected = self.speaker_project_mel(self.speaker_embeddings)

    def compute_gst(self, inputs, mel_specs):
        # TODO: gst compute
        #gst_outputs = self.gst_layer(mel_specs)
        #inputs = self._add_speaker_embedding(inputs, gst_outputs)
        return inputs

    def call(self, characters, text_lengths, mel_specs, speaker_ids=None):
        B = characters.shape[0]
        mask = tf.sequence_mask(text_lengths)
        inputs = self.embedding(characters)
        self._init_states()
        # self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            inputs = self._concat_speaker_embedding(inputs,
                                                    self.speaker_embeddings)
        encoder_outputs = self.encoder(inputs)
        if self.gst:
            encoder_outputs = self.compute_gst(encoder_outputs, mel_specs)
        if self.num_speakers > 1:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs)
        mel_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask)
        mel_outputs = tf.reshape(mel_outputs, [B, -1, self.mel_dim])
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def inference(self, characters, speaker_ids=None, style_mel=None):
        B = characters.shape[0]
        inputs = self.embedding(characters)
        self._init_states()
        self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            inputs = self._concat_speaker_embedding(inputs,
                                                    self.speaker_embeddings)
        encoder_outputs = self.encoder(inputs)
        if self.gst and style_mel is not None:
            encoder_outputs = self.compute_gst(encoder_outputs, style_mel)
        if self.num_speakers > 1:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs, self.speaker_embeddings)
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs, self.speaker_embeddings_projected)
        mel_outputs = tf.reshape(mel_outputs, [B, -1, self.mel_dim])
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def _compute_speaker_embedding(self, speaker_ids):
        speaker_embeddings = self.speaker_embedding(speaker_ids)
        return speaker_embeddings

    @staticmethod
    def _add_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.shape[0], outputs.shape[1], -1)
        outputs = outputs + speaker_embeddings_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = tf.tile(tf.expand_dims(speaker_embeddings, 1), [1, outputs.shape[1], 1])
        outputs = tf.concat([outputs, speaker_embeddings_], axis=-1)
        return outputs

