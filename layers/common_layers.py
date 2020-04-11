import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow_addons.seq2seq import BahdanauAttention

from TTS_tf.utils.tf_utils import shape_list


class Linear(keras.layers.Layer):
    def __init__(self, units, use_bias=True):
        super().__init__()
        self.linear_layer = keras.layers.Dense(units, use_bias=use_bias)
        self.act = keras.layers.ReLU()

    def call(self, x):
        """
        shapes:
            x: B x T x C
        """
        return self.act(self.linear_layer(x))


class LinearBN(keras.layers.Layer):
    def __init__(self, units, use_bias=True):
        super().__init__()
        self.linear_layer = keras.layers.Dense(units, use_bias=use_bias)
        self.bn = keras.layers.BatchNormalization(axis=-1)
        self.act = keras.layers.ReLU()

    def call(self, x):
        """
        shapes:
            x: B x T x C
        """
        out = self.linear_layer(x)
        out = self.bn(out)
        return self.act(out)


class Prenet(keras.layers.Layer):
    def __init__(self,
                 prenet_type="original",
                 prenet_dropout=True,
                 units=[256, 256],
                 bias=True):
        super().__init__()
        self.prenet_type = prenet_type
        self.prenet_dropout = prenet_dropout
        self.linear_layers = []
        if prenet_type == "bn":
            self.linear_layers += [LinearBN(unit, use_bias=bias) for unit in units]
        elif prenet_type == "original":
            self.linear_layers += [Linear(unit, use_bias=bias) for unit in units]
        if prenet_dropout:
            self.dropout = keras.layers.Dropout(rate=0.5)

    def call(self, x):
        """
        shapes:
            x: B x T x C
        """
        for linear in self.linear_layers:
            if self.prenet_dropout:
                x = self.dropout(linear(x))
            else:
                x = linear(x)
        return x



def _location_sensitive_score(processed_query, keys, processed_loc, attention_v, attention_b):
    dtype = processed_query.dtype
    num_units = keys.shape[-1].value or array_ops.shape(keys)[-1]
    return tf.reduce_sum(attention_v * tf.tanh(keys + processed_query + processed_loc + attention_b), [2])


class LocationSensitiveAttention(BahdanauAttention):
    def __init__(self,
                 units,
                 memory=None,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn="softmax",
                 kernel_initializer="glorot_uniform",
                 dtype=None,
                 name="LocationSensitiveAttention",
                 location_attention_filters=32,
                 location_attention_kernel_size=31):

        super(LocationSensitiveAttention,
                    self).__init__(units=units,
                                    memory=memory,
                                    memory_sequence_length=memory_sequence_length,
                                    normalize=normalize,
                                    probability_fn='softmax',  ## parent module default
                                    kernel_initializer=kernel_initializer,
                                    dtype=dtype,
                                    name=name)
        if probability_fn == 'sigmoid':
            self.probability_fn = lambda score, _: self._sigmoid_normalization(score)
        self.location_conv = keras.layers.Conv1D(filters=location_attention_filters, kernel_size=location_attention_kernel_size, padding='same', use_bias=False)
        self.location_dense = keras.layers.Dense(units, use_bias=False)
        # self.v = keras.layers.Dense(1, use_bias=True)
        
    def  _location_sensitive_score(self, processed_query, keys, processed_loc):
        processed_query = tf.expand_dims(processed_query, 1)
        return tf.reduce_sum(self.attention_v * tf.tanh(keys + processed_query + processed_loc), [2])

    def _location_sensitive(self, alignment_cum, alignment_old):
        alignment_cat = tf.stack([alignment_cum, alignment_old], axis=2)
        return self.location_dense(self.location_conv(alignment_cat))

    def _sigmoid_normalization(self, score):
        return tf.nn.sigmoid(score) / tf.reduce_sum(tf.nn.sigmoid(score), axis=-1, keepdims=True)

    # def _apply_masking(self, score, mask):
    #     padding_mask = tf.expand_dims(math_ops.logical_not(mask), 2)
    #     # Bias so padding positions do not contribute to attention distribution.
    #     score -= 1.e9 * math_ops.cast(padding_mask, dtype=tf.float32)
    #     return score

    def _calculate_attention(self, query, state):
        alignment_cum, alignment_old = state[:2]
        processed_query = self.query_layer(
            query) if self.query_layer else query
        processed_loc = self._location_sensitive(alignment_cum, alignment_old)
        score = self._location_sensitive_score(
            processed_query,
            self.keys,
            processed_loc)
        alignment = self.probability_fn(score, state)
        alignment_cum = alignment_cum + alignment
        state[0] = alignment_cum
        state[1] = alignment
        return alignment, state

    def compute_context(self, alignments):
        expanded_alignments = tf.expand_dims(alignments, 1)
        context = tf.matmul(expanded_alignments, self.values)
        context = tf.squeeze(context, [1])
        return context

    # def call(self, query, state):
    #     alignment, next_state = self._calculate_attention(query, state)
    #     return alignment, next_state


class Attention(keras.layers.Layer):
    """TODO: implement forward_attention"""
    """TODO: location sensitive attention"""
    """TODO: implement attention windowing """
    def __init__(self, attn_dim, use_loc_attn, loc_attn_n_filters,
                 loc_attn_kernel_size, use_windowing, norm, use_forward_attn,
                 use_trans_agent, use_forward_attn_mask):
        super(Attention, self).__init__()
        self.use_loc_attn = use_loc_attn
        self.loc_attn_n_filters = loc_attn_n_filters
        self.loc_attn_kernel_size = loc_attn_kernel_size
        self.use_windowing = use_windowing
        self.norm = norm
        self.use_forward_attn = use_forward_attn
        self.use_trans_agent = use_trans_agent
        self.use_forward_attn_mask = use_forward_attn_mask
        self.query_layer = tf.keras.layers.Dense(attn_dim, use_bias=False)
        self.input_layer = tf.keras.layers.Dense(attn_dim, use_bias=False)
        self.v = tf.keras.layers.Dense(1, use_bias=True)
        if use_loc_attn:
            self.loc_conv = keras.layers.Conv1D(
                filters=loc_attn_n_filters,
                kernel_size=loc_attn_kernel_size,
                padding='same',
                use_bias=False)
            self.loc_dense = keras.layers.Dense(attn_dim, use_bias=False)

    def init_states(self, values):
        pass

    def process_values(self, values):
        """ preprocess values since this compution is repeating each
        decoder step """
        self.processed_values = self.input_layer(values)

    def get_loc_attn(self, query):
        attn_cat = tf.concat([self.attn_weights, self.attn_weights_cum],
                             axis=2)
        processed_query = self.query_layer(tf.expand_dims(query, 1))
        processed_attn = self.loc_dense(self.loc_conv(attn_cat))
        score = self.v(
            tf.nn.tanh(self.processed_values + processed_query +
                       processed_attn))
        return score, processed_query

    def get_attn(self, query):
        """ compute query layer and unnormalized attention weights """
        processed_query = self.query_layer(tf.expand_dims(query, 1))
        score = self.v(tf.nn.tanh(self.processed_values + processed_query))
        return score, processed_query

    def apply_score_masking(self, score, mask):
        padding_mask = tf.expand_dims(math_ops.logical_not(mask), 2)
        # Bias so padding positions do not contribute to attention distribution.
        score -= 1.e9 * math_ops.cast(padding_mask, dtype=tf.float32)
        return score

    def call(self, query, values, mask=None):
        """
        shapes:
            query: B x D
            values: B x T x D
            contect_vec: B x D
            attn_weights: B x T
        """
        assert self.processed_values is not None, " [!] make sure calling process_values() before running attention"
        score, processed_query = self.get_attn(query)
        # masking
        if mask is not None:
            self.apply_score_masking(score, mask)
        # attn_weights shape == (batch_size, max_length, 1)
        # TODO: implement softmax
        attn_weights = tf.nn.sigmoid(score)
        attn_weights = attn_weights / tf.reduce_sum(
            attn_weights, axis=1, keepdims=True)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.matmul(attn_weights, values, transpose_a=True, transpose_b=False)
        # context_vector = tf.squeeze(context_vector, axis=1)
        # context_vector = attn_weights * values
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        self.attn_weights = attn_weights
        return context_vector
