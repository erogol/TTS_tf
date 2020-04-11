
import tensorflow as tf
from tensorflow import keras
from TTS_tf.utils.tf_utils import shape_list
from TTS_tf.layers.common_layers import Prenet, LocationSensitiveAttention
from tensorflow_addons.seq2seq import AttentionWrapper


class ConvBNBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=None):
        super(ConvBNBlock, self).__init__()
        self.conv1d = keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.norm = keras.layers.BatchNormalization(axis=2, momentum=0.99, epsilon=1e-3)
        self.dropout = keras.layers.Dropout(rate=0.5)
        self.act = keras.layers.Activation(activation)

    def call(self, x):
        o = self.conv1d(x)
        o = self.norm(o)
        o = self.act(o)
        o = self.dropout(o)
        return o


class Postnet(keras.layers.Layer):
    def __init__(self, output_filters, num_convs=5):
        super(Postnet, self).__init__()
        self.convolutions = []
        self.convolutions.append(ConvBNBlock(512, 5, 'tanh'))
        for _ in range(1, num_convs - 1):
            self.convolutions.append(ConvBNBlock(512, 5, 'tanh'))
        self.convolutions.append(ConvBNBlock(output_filters, 5, 'linear'))

    def call(self, x):
        o = x
        for layer in self.convolutions:
            o = layer(o)
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, filters=512):
        super(Encoder, self).__init__()
        self.convolutions = []
        for _ in range(3):
            self.convolutions.append(ConvBNBlock(filters, 5, 'relu'))
        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(filters // 2, return_sequences=True))

    def call(self, x):
        o = x
        for layer in self.convolutions:
            o = layer(o)
        o = self.lstm(o)
        return o


class Decoder(keras.layers.Layer):
    def __init__(self, frame_dim, r, attn_type, use_attn_win, attn_norm, prenet_type,
                 prenet_dropout, use_forward_attn, use_trans_agent, use_forward_attn_mask,
                 use_location_attn, attn_K, separate_stopnet, speaker_emb_dim):
        super(Decoder, self).__init__()
        self.frame_dim = frame_dim
        self.r_init = tf.constant(r, dtype=tf.int32)
        self.r = tf.constant(r, dtype=tf.int32)
        self.separate_stopnet = separate_stopnet
        self.max_decoder_steps = tf.constant(1000, dtype=tf.int32)
        self.stop_thresh = tf.constant(0.5, dtype=tf.float32)

        # model dimensions
        self.query_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.attn_dim = 128
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        self.prenet = Prenet(prenet_type, prenet_dropout, [self.prenet_dim,
                                                           self.prenet_dim], bias=False)
        self.attention_rnn = keras.layers.LSTMCell(self.query_dim, name='attention_rnn')
        self.attention_rnn_dropout = keras.layers.Dropout(0.5)
        # TODO: implement other attn options
        self.attention = LocationSensitiveAttention(units=self.attn_dim,
                                                    normalize=False,
                                                    probability_fn=attn_norm)
        self.attention_mechanism = AttentionWrapper(self.attention_rnn,
                                                    self.attention)
        self.decoder_rnn = keras.layers.LSTMCell(self.decoder_rnn_dim, name='decoder_rnn')
        self.decoder_rnn_dropout = keras.layers.Dropout(0.5)
        self.linear_projection = keras.layers.Dense(self.frame_dim * r)
        self.stopnet = keras.layers.Dense(1)

    def set_r(self, new_r):
        self.r = tf.constant(new_r, dtype=tf.int32)

    def build_decoder_initial_states(self, batch_size, memory_dim, memory_length):
        zero_frame = tf.zeros([batch_size, 1, self.frame_dim])
        zero_context = tf.zeros([batch_size, memory_dim])
        attention_rnn_state = self.attention_rnn.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        decoder_rnn_state = self.decoder_rnn.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        alignment_cum = tf.zeros([batch_size, memory_length])
        alignment_old = tf.zeros([batch_size, memory_length])
        attention_states = [alignment_cum, alignment_old]
        return zero_frame, zero_context, attention_rnn_state, decoder_rnn_state, \
                attention_states

    def step(self, prenet_next, states,
             memory_seq_length=None):
        _, context_next, attention_rnn_state, decoder_rnn_state, \
                attention_states = states
        attention_rnn_input = tf.concat([prenet_next, context_next], -1)
        attention_rnn_output, attention_rnn_state = \
                self.attention_rnn(attention_rnn_input,
                                   attention_rnn_state)
        attention_rnn_output = self.attention_rnn_dropout(attention_rnn_output)
        attention, attention_states = self.attention([attention_rnn_output, attention_states])
        context = self.attention.compute_context(attention)
        decoder_rnn_input = tf.concat([attention_rnn_output, context], -1)
        decoder_rnn_output, decoder_rnn_state = \
                self.decoder_rnn(decoder_rnn_input, decoder_rnn_state)
        decoder_rnn_output = self.decoder_rnn_dropout(decoder_rnn_output)
        linear_projection_input = tf.concat([decoder_rnn_output, context], -1)
        output_frame = self.linear_projection(linear_projection_input)
        stopnet_input = tf.concat([decoder_rnn_output, output_frame], -1)
        stopnet_output = self.stopnet(stopnet_input)
        output_frame = output_frame[:, :self.r * self.frame_dim]
        states = (states[0], context, attention_rnn_state, decoder_rnn_state, attention_states)
        return output_frame, stopnet_output, states, attention

    def decode(self, memory, frames, states, memory_seq_length=None):
        B, T, D = shape_list(memory)
        num_iter = shape_list(frames)[1] // self
        # init states
        frame_zero = states[0]
        frames = tf.concat([frame_zero, frames], axis=1)
        outputs = tf.TensorArray(dtype=tf.float32, size=num_iter)
        attentions = tf.TensorArray(dtype=tf.float32, size=num_iter)
        stop_tokens = tf.TensorArray(dtype=tf.float32, size=num_iter)
        # pre-computes
        self.attention.setup_memory(memory)
        prenet_output = self.prenet(frames)
        step_count = tf.constant(0, dtype=tf.int32) 

        def _body(step, memory, prenet_output, states, outputs, stop_tokens, attentions):
            prenet_next = prenet_output[:, step]
            states_old = states
            output, stop_token, states, attention = self.step(prenet_next,
                                                              states,
                                                              memory_seq_length)
            outputs = outputs.write(step, output)
            attentions = attentions.write(step, attention)
            stop_tokens = stop_tokens.write(step, stop_token)
            return step + 1, memory, prenet_output, states, outputs, stop_tokens, attentions
        _, memory, _, states, outputs, stop_tokens, attentions = \
                tf.while_loop(lambda *arg: True,
                    _body,
                    loop_vars=(step_count, memory, prenet_output, states, outputs,
                               stop_tokens, attentions),
                    parallel_iterations=32,
                    swap_memory=True,
                    maximum_iterations=num_iter)

        outputs = outputs.stack()
        attentions = attentions.stack()
        stop_tokens = stop_tokens.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
        attentions = tf.transpose(attentions, [1, 0 ,2])
        stop_tokens = tf.transpose(stop_tokens, [1, 0, 2])
        stop_tokens = tf.squeeze(stop_tokens, axis=2)
        return outputs, stop_tokens, attentions

    def call(self, memory, frames, states, memory_seq_length=None):
        return self.decode(memory, frames, states, memory_seq_length)
